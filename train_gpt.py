import dataclasses
import glob
import math
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.nn import dot_product_attention as dpa
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from flax.jax_utils import prefetch_to_device

# ------------------------------
# Data
# ------------------------------

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        raise ValueError("magic number mismatch in the data .bin file")
    assert header[1] == 1, "unsupported version"
    return int(header[2])


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header"
    return tokens


class DataLoader:
    def __init__(self, filename_pattern: str, B: int, T: int):
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert self.files, f"no files match pattern: {filename_pattern}"
        total = 0
        for f in self.files:
            shard_ntok = _peek_data_shard(f)
            assert shard_ntok >= B * T + 1
            total += shard_ntok
        print(f"DataLoader: {total:,} tokens across {len(self.files)} shards")
        self.current_shard = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = 0

    def reset(self):
        self.current_shard = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = 0

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x_BL = np.array(buf[:-1].astype(np.int32)).reshape(B, T)
        y_BL = np.array(buf[1:].astype(np.int32)).reshape(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.advance()
        return x_BL, y_BL

def data_generator(loader: DataLoader, Ng: int, Nd: int, B_local: int, T: int):
    """A generator that yields batches for training, pre-shaped for pmap."""
    while True:
        x_stack, y_stack = [], []
        for _ in range(Ng):
            x_BL, y_BL = loader.next_batch()
            x_stack.append(x_BL)
            y_stack.append(y_BL)
        
        x_NgBL = np.stack(x_stack, axis=0)
        y_NgBL = np.stack(y_stack, axis=0)
        
        x_NgNdBL = x_NgBL.reshape(Ng, Nd, B_local, T)
        y_NgNdBL = y_NgBL.reshape(Ng, Nd, B_local, T)
        
        x_NdNgBL = np.swapaxes(x_NgNdBL, 0, 1)
        y_NdNgBL = np.swapaxes(y_NgNdBL, 0, 1)
        
        yield x_NdNgBL, y_NdNgBL


# ------------------------------
# Model
# ------------------------------

@dataclasses.dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    kernel_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)
    embed_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)
    residual_init: nn.initializers.Initializer | None = None
    dtype: jnp.dtype = jnp.float32

@dataclasses.dataclass
class Hyperparameters:
    input_bin: str = "fineweb10B/fineweb_train_*.bin"
    input_val_bin: str = "fineweb10B/fineweb_val_*.bin"
    val_loss_every: int = 250
    batch_size: int = 32
    sequence_length: int = 1024
    total_batch_size: int = 524288
    learning_rate: float = 6e-4
    warmup_iters: int = 700
    learning_rate_decay_frac: float = 0.0
    weight_decay: float = 0.1
    num_iterations: int = 18865
    overfit_single_batch: bool = False
    dtype: str = "bfloat16"

class MLP(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, x_BLD):
        x_BLD = nn.Dense(4 * self.cfg.n_embd, kernel_init=self.cfg.kernel_init, use_bias=True, dtype=self.cfg.dtype)(x_BLD)
        x_BLD = 0.5 * x_BLD * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x_BLD + 0.044715 * jnp.power(x_BLD, 3.0))))
        residual_init = self.cfg.residual_init or self.cfg.kernel_init
        x_BLD = nn.Dense(self.cfg.n_embd, kernel_init=residual_init, use_bias=True, dtype=self.cfg.dtype)(x_BLD)
        return x_BLD

class CausalSelfAttention(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, x_BLD):
        B, L, D = x_BLD.shape
        assert D == self.cfg.n_embd
        qkv_BL3D = nn.Dense(3 * self.cfg.n_embd, kernel_init=self.cfg.kernel_init, use_bias=True, dtype=self.cfg.dtype)(x_BLD)
        q_BLD, k_BLD, v_BLD = jnp.split(qkv_BL3D, 3, axis=2)
        Dh = self.cfg.n_embd // self.cfg.n_head
        q_BLHDh = q_BLD.reshape(B, L, self.cfg.n_head, Dh)
        k_BLHDh = k_BLD.reshape(B, L, self.cfg.n_head, Dh)
        v_BLHDh = v_BLD.reshape(B, L, self.cfg.n_head, Dh)
        causal_LL = jnp.tril(jnp.ones((L, L), dtype=bool))
        mask_BHLL = jnp.broadcast_to(causal_LL, (B, self.cfg.n_head, L, L))
        y_BLHDh = dpa(q_BLHDh, k_BLHDh, v_BLHDh, mask=mask_BHLL, implementation="cudnn")
        y_BLD = y_BLHDh.reshape(B, L, D)
        residual_init = self.cfg.residual_init or self.cfg.kernel_init
        y_BLD = nn.Dense(self.cfg.n_embd, kernel_init=residual_init, use_bias=True, dtype=self.cfg.dtype)(y_BLD)
        return y_BLD

class Block(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, x_BLD):
        x_BLD = x_BLD + CausalSelfAttention(self.cfg)(nn.LayerNorm(use_bias=True, epsilon=1e-5, dtype=jnp.float32)(x_BLD))
        x_BLD = x_BLD + MLP(self.cfg)(nn.LayerNorm(use_bias=True, epsilon=1e-5, dtype=jnp.float32)(x_BLD))
        return x_BLD

class GPT(nn.Module):
    cfg: Config
    def setup(self):
        cfg = self.cfg
        self.wte = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.n_embd, embedding_init=cfg.embed_init)
        self.wpe = nn.Embed(num_embeddings=cfg.block_size, features=cfg.n_embd, embedding_init=cfg.embed_init)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(dtype=jnp.float32, use_bias=True, epsilon=1e-5)

    def __call__(self, idx_BL, targets_BL=None):
        _, L = idx_BL.shape
        assert L <= self.cfg.block_size
        pos_L = jnp.arange(0, L, dtype=jnp.int32)
        tok_emb_BLD = self.wte(idx_BL)
        pos_emb_LD = self.wpe(pos_L)
        x_BLD = (tok_emb_BLD + pos_emb_LD).astype(self.cfg.dtype)
        for block in self.blocks:
            x_BLD = block(x_BLD)
        x_BLD = self.ln_f(x_BLD)
        if targets_BL is not None:
            logits_BLV = jnp.matmul(x_BLD, self.wte.embedding.T)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_BLV.reshape(-1, logits_BLV.shape[-1]).astype(jnp.float32),
                targets_BL.reshape(-1),
            ).mean()
            return logits_BLV, loss
        else:
            logits_B1V = jnp.matmul(x_BLD[:, -1:, :], self.wte.embedding.T)
            return logits_B1V, None


# ------------------------------
# Train utils
# ------------------------------
def create_train_state(key: jax.Array, cfg: Config, lr_schedule: optax.Schedule, weight_decay: float) -> train_state.TrainState:
    model = GPT(cfg)
    dummy_input = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
    dummy_targets = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
    params = model.init(key, dummy_input, dummy_targets)["params"]
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")
    decay_mask = jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "decay": optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=0.9, b2=0.95),
                "no_decay": optax.adamw(learning_rate=lr_schedule, weight_decay=0.0, b1=0.9, b2=0.95),
            },
            jax.tree_util.tree_map(lambda d: "decay" if d else "no_decay", decay_mask),
        ),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.pmap, axis_name="batch")
def eval_step(state: train_state.TrainState, inputs_BL: jax.Array, labels_BL: jax.Array):
    _, loss = state.apply_fn({"params": state.params}, inputs_BL, labels_BL)
    return loss

def make_train_step(Ng: int):
    @partial(jax.pmap, axis_name="batch")
    def train_step(state: train_state.TrainState, inputs_NgBL: jax.Array, labels_NgBL: jax.Array):
        def grad_accumulation_body(carry, mb):
            state_in = carry
            inputs_BL, labels_BL = mb
            def compute_loss(params):
                _, loss = state_in.apply_fn({"params": params}, inputs_BL, labels_BL)
                return loss
            loss, grads = jax.value_and_grad(compute_loss)(state_in.params)
            return carry, (grads, loss)

        _, (grads_seq, loss_seq) = jax.lax.scan(
            grad_accumulation_body, state, (inputs_NgBL, labels_NgBL), length=Ng
        )

        grads_avg_local = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads_seq)
        loss_avg_local = jnp.mean(loss_seq)
        grads_avg = jax.lax.pmean(grads_avg_local, axis_name="batch")
        loss_avg = jax.lax.pmean(loss_avg_local, axis_name="batch")
        
        new_state = state.apply_gradients(grads=grads_avg)
        
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads_avg))
        )
        
        return new_state, loss_avg, grad_norm
    return train_step
        
# ------------------------------
# Main
# ------------------------------
def main():
    h = Hyperparameters()

    print(f"JAX devices: {jax.devices()}")
    Nd = jax.device_count()

    batch_size = h.batch_size
    T = h.sequence_length
    total_batch_size = h.total_batch_size
    lr = h.learning_rate
    warmup_iters = h.warmup_iters
    lr_decay_frac = h.learning_rate_decay_frac
    weight_decay = h.weight_decay
    num_iterations = h.num_iterations
    val_loss_every = h.val_loss_every
    val_max_steps = 20
    overfit_single_batch = bool(h.overfit_single_batch)

    residual_std = 0.02 / math.sqrt(2 * 12)
    dtype = jnp.float32 if h.dtype == "float32" else jnp.bfloat16
    cfg = Config(
        block_size=T, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
        residual_init=nn.initializers.normal(stddev=residual_std), dtype=dtype,
    )

    global_batch_size = batch_size * Nd
    train_loader = DataLoader(h.input_bin, global_batch_size, T)
    val_loader = DataLoader(h.input_val_bin, global_batch_size, T) if h.input_val_bin else None

    tokens_per_fwdbwd = batch_size * T * Nd
    assert total_batch_size % tokens_per_fwdbwd == 0
    Ng = total_batch_size // tokens_per_fwdbwd
    print(f"tokens per fwd/bwd (global): {tokens_per_fwdbwd}")
    print(f"grad accumulation steps (Ng): {Ng}")

    train_iterator = data_generator(
        loader=train_loader, Ng=Ng, Nd=Nd, B_local=batch_size, T=T
    )
    train_iterator = prefetch_to_device(train_iterator, size=1)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_iters,
        decay_steps=max(1, num_iterations - warmup_iters), end_value=lr * lr_decay_frac,
    )

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    state = create_train_state(init_key, cfg, lr_schedule, weight_decay)
    state = jax.device_put_replicated(state, jax.local_devices())

    train_step = make_train_step(Ng)

    print("Start Training")
    for step in range(num_iterations + 1):
        last_step = step == num_iterations

        if val_loss_every > 0 and (step % val_loss_every == 0 or last_step) and val_loader is not None:
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_max_steps):
                x_BL, y_BL = val_loader.next_batch()
                inputs_NdBL = x_BL.reshape((Nd, batch_size, T))
                labels_NdBL = y_BL.reshape((Nd, batch_size, T))
                loss = eval_step(state, inputs_NdBL, labels_NdBL)
                val_loss += jnp.mean(loss)
            val_loss /= val_max_steps
            print(f"val loss {float(val_loss):.6f}")

        if last_step:
            break

        if overfit_single_batch:
            train_loader.reset()
            train_iterator = data_generator(
                loader=train_loader, Ng=Ng, Nd=Nd, B_local=batch_size, T=T
            )
            train_iterator = prefetch_to_device(train_iterator, size=1)

        start_time = time.time()
        x_NdNgBL, y_NdNgBL = next(train_iterator)
        
        state, loss, grad_norm = train_step(state, x_NdNgBL, y_NdNgBL)
        
        grad_norm.block_until_ready()
        end_time = time.time()
        step_time_ms = (end_time - start_time) * 1000

        cur_lr = float(lr_schedule(step))
        print(
            f"step {step+1:4d}/{num_iterations} | train loss {float(jnp.mean(loss)):.6f} | "
            f"lr {cur_lr:.2e} | norm {float(jnp.mean(grad_norm)):.2f} | "
            f"step_time {step_time_ms:.2f}ms"
        )

if __name__ == "__main__":
    main()