from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import dataclasses
import optax

@dataclasses.dataclass
class Config:
  D: int  # Embed dimension
  H: int  # num of heads
  L: int  # seq length
  N: int  # num of layers
  V: int  # vocab size
  F: int  # ffn inner dimension

  dtype: jnp.dtype = jnp.float32

class MLP(nn.Module):
  cfg: Config

  @nn.compact
  def __call__(self, x_BLD: jax.Array):
    x_BLF = nn.Dense(self.cfg.F, use_bias=False)(x_BLD)
    x_BLF = nn.gelu(x_BLF)
    x_BLD = nn.Dense(self.cfg.D, use_bias=False)(x_BLF)

    return x_BLD

class SelfAttention(nn.Module):
  cfg: Config

  @nn.compact
  def __call__(self, x_BLD: jax.Array):
    cfg = self.cfg

    Dh = cfg.D // cfg.H # dims per head

    multilinear = partial(nn.DenseGeneral, axis=-1, features=(cfg.H, Dh), use_bias=False, dtype=cfg.dtype)

    q_BLHDh, k_BLHDh, v_BLHDh = (
        multilinear(name='query')(x_BLD),
        multilinear(name='key')(x_BLD),
        multilinear(name='value')(x_BLD)
    )

    q_BLHDh *= Dh ** 0.5

    attn_BHLL = jnp.einsum('...qhd,...khd->...hqk', q_BLHDh, k_BLHDh)
    # cast to fp32 for softmax (exp why?)
    attn_BHLL = attn_BHLL.astype(jnp.float32)

    L = x_BLD.shape[1]
    mask_11LL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

    _NEG_INF = jnp.finfo(cfg.dtype).min
    attn_BHLL = jnp.where(mask_11LL, attn_BHLL, _NEG_INF)
    attn_BHLL = nn.softmax(attn_BHLL, axis=-1)
    attn_BHLL = attn_BHLL.astype(cfg.dtype)
    out_BLHDh = jnp.einsum('...hqk,...khd->...qhd', attn_BHLL, v_BLHDh)

    return nn.DenseGeneral(axis=(-2,-1), features=cfg.D, name='attn_out_proj', use_bias=False, dtype=cfg.dtype)(out_BLHDh)

class Block(nn.Module):
  cfg: Config

  @nn.compact
  def __call__(self, in_BLD: jax.Array):
    x_BLD = nn.LayerNorm(use_bias=False, dtype=self.cfg.dtype)(in_BLD)
    x_BLD = SelfAttention(self.cfg)(x_BLD)
    x_BLD += in_BLD

    z_BLD = nn.LayerNorm(use_bias=False, dtype=self.cfg.dtype)(x_BLD)
    z_BLD = MLP(self.cfg)(x_BLD)

    return x_BLD + z_BLD

class Transformer(nn.Module):
  cfg: Config

  def setup(self):
    cfg = self.cfg

    self.embed = nn.Embed(num_embeddings=cfg.V, features=cfg.D)
    self.pos_embed = nn.Embed(num_embeddings=cfg.L, features=cfg.D)

    self.blocks = [Block(cfg) for _ in range(cfg.N)]

    self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)

  
  def __call__(self, y_BL: jax.Array):
    y_BLD = self.embed(y_BL)
    y_BLD += self.pos_embed(jnp.arange(0, y_BL.shape[1])[None, ...])

    for block in self.blocks:
      y_BLD = block(y_BLD)
    
    y_BLD = self.out_ln(y_BLD)
    return self.embed.attend(y_BLD.astype(jnp.float32))


def create_train_state(key: jax.Array, cfg: Config, learning_rate: float) -> train_state.TrainState:
  model = Transformer(cfg)

  dummy_input = jnp.ones((1, cfg.L), dtype=jnp.int32)
  params = model.init(key, dummy_input)['params']

  tx = optax.adamw(learning_rate)

  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def loss_fn(logits, labels):
  return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

@jax.jit
def train_step(state: train_state.TrainState, batch: jax.Array):
  inputs = batch[:, :-1]
  labels = batch[:, 1:]

  def compute_loss(params):
    logits = state.apply_fn({'params': params}, inputs)
    return loss_fn(logits, labels)
  
  loss, grads = jax.value_and_grad(compute_loss)(state.params)
  state = state.apply_gradients(grads=grads)

  return state, loss

#if __name__ == 'main':
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
TRAINING_STEPS = 1000

cfg = Config(
    D=256,
    H=4,
    L=128,
    N=4,
    V=1000,
    F=4 * 256
)

key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)

state = create_train_state(init_key, cfg, LEARNING_RATE)

print(jax.default_backend())   # 'cpu', 'gpu', or 'tpu'
print(jax.devices())        # Device(id=0, process_index=0, platform='gpu')
print("Start Training")
for step in range(TRAINING_STEPS):
  key, data_key = jax.random.split(key)

  dummy_batch = jax.random.randint(
      key=data_key,
      shape=(BATCH_SIZE, cfg.L),
      minval=0,
      maxval=cfg.V
  )

  state, loss = train_step(state, dummy_batch)
  if step % 100 == 0:
    print(f"Step: {step}, Loss: {loss}")
