from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import dataclasses
import optax
import glob
import numpy as np
from flax.jax_utils import replicate
import os


class DataLoader:
  def __init__(self, batch_size, seq_len, split="train", data_dir="fineweb10B"):
    self.B = batch_size
    self.L = seq_len
    self.data_dir = data_dir

    train_files = sorted(glob.glob(os.path.join(self.data_dir, f"fineweb_{split}_*.bin")))
    if not train_files:
      raise FileNotFoundError(
          f"No {split} data found in {self.data_dir}"
      )
    
    self.train_chunks = [np.memmap(f, dtype=np.uint16, mode='r') for f in train_files]
    

  def __iter__(self):
    while True:
      chunk_idx = np.random.randint(0, len(self.train_chunks))
      chunk = self.train_chunks[chunk_idx]

      starts = np.random.randint(0, len(chunk) - self.L, size=self.B) 
      x = np.array([chunk[s : s + self.L] for s in starts])
      yield x


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

    q_BHLDh = jnp.transpose(q_BLHDh, (0, 2, 1, 3))
    k_BHLDh = jnp.transpose(k_BLHDh, (0, 2, 1, 3))
    v_BHLDh = jnp.transpose(v_BLHDh, (0, 2, 1, 3))

    out_BHLDh = jax.nn.dot_product_attention(
        q_BHLDh,
        k_BHLDh,
        v_BHLDh,
        bias=None,
        mask=None,
        is_causal=True,
    )
    out_BLHDh = jnp.transpose(out_BHLDh, (0, 2, 1, 3))

    return nn.DenseGeneral(axis=(-2,-1), features=cfg.D, name='attn_out_proj', use_bias=False, dtype=cfg.dtype)(out_BLHDh)

class Block(nn.Module):
  cfg: Config

  @nn.compact
  def __call__(self, in_BLD: jax.Array):
    x_BLD = nn.LayerNorm(use_bias=False, dtype=self.cfg.dtype)(in_BLD)
    x_BLD = SelfAttention(self.cfg)(x_BLD)
    x_BLD += in_BLD

    z_BLD = nn.LayerNorm(use_bias=False, dtype=self.cfg.dtype)(x_BLD)
    z_BLD = MLP(self.cfg)(z_BLD)

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

#@jax.jit
@partial(jax.pmap, axis_name='batch')
def train_step(state: train_state.TrainState, batch: jax.Array):
  inputs = batch[:, :-1]
  labels = batch[:, 1:]

  def compute_loss(params):
    logits = state.apply_fn({'params': params}, inputs)
    return loss_fn(logits, labels)
  
  loss, grads = jax.value_and_grad(compute_loss)(state.params)

  grads = jax.lax.pmean(grads, axis_name='batch')
  loss = jax.lax.pmean(loss, axis_name='batch')

  state = state.apply_gradients(grads=grads)

  return state, loss

@partial(jax.pmap, axis_name='batch')
def eval_step(state: train_state.TrainState, batch: jax.Array):
  inputs = batch[:, :-1]
  labels = batch[:, 1:]

  logits = state.apply_fn({'params': state.params}, inputs)
  loss = loss_fn(logits, labels)

  return loss


@partial(jax.pmap, axis_name='batch')
def average_across_devices(x):
  return jax.lax.pmean(x, axis_name='batch')

if __name__ == '__main__':
  BATCH_SIZE = 8
  LEARNING_RATE = 3e-4
  TRAINING_STEPS = 10000
  VAL_EVERY = 125
  VAL_STEPS = 10
  
  
  cfg = Config(
      D=768,
      H=12,
      L=1024,
      N=12,
      V=50257,
      F=4 * 768
  )
  
  num_devices = jax.local_device_count()
  
  dataloader = DataLoader(num_devices * BATCH_SIZE, cfg.L)
  data_iter = iter(dataloader)
  
  val_dataloader = DataLoader(num_devices * BATCH_SIZE, cfg.L, split="val")
  val_data_iter = iter(val_dataloader)
  
  key = jax.random.PRNGKey(0)
  key, init_key = jax.random.split(key)
  
  state = create_train_state(init_key, cfg, LEARNING_RATE)
  state = replicate(state)
  
  print("Start Training")
  for step in range(TRAINING_STEPS):
    key, data_key = jax.random.split(key)
  
    batch = jnp.asarray(next(data_iter))
    sharded_batch = batch.reshape(num_devices, -1, batch.shape[-1])
  
    state, loss = train_step(state, sharded_batch)
    if step % 25 == 0:
      print(f"Step: {step}, Loss: {loss[0]:.4f}")
    
    if step > 0 and (step % VAL_EVERY == 0 or step == TRAINING_STEPS - 1):
      val_loss_accumulator = jnp.zeros(())
  
      for _ in range(VAL_STEPS):
        val_batch = jnp.asarray(next(val_data_iter))
        sharded_val_batch = val_batch.reshape(num_devices, -1, val_batch.shape[-1])
  
        loss_per_device = eval_step(state, sharded_val_batch)
        val_loss_accumulator += loss_per_device
    
      val_loss_accumulator /= VAL_STEPS
      final_val_loss = average_across_devices(val_loss_accumulator)
  
      print(f"Step: {step}, Val Loss: {final_val_loss[0]:.4f}")
