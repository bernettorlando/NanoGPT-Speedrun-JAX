# JAX/Flax NanoGPT

This repository is a JAX/Flax implementation of a GPT-style model, inspired by [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). It is designed to be a simple and clear example of how to train a transformer language model using JAX.

## Installation

1.  **Install JAX with CUDA support.**

    For CUDA 12, run the following command:

    ```bash
    python -m pip install --upgrade --no-cache-dir --force-reinstall \
      --index-url https://pypi.org/simple \
      --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
      "jax[cuda12]"
    ```

    For other CUDA or CPU/TPU versions, please refer to the [official JAX installation guide](https://github.com/google/jax#installation).

2.  **Install other dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Download the dataset.**

    This project trains on the FineWeb 10B dataset. Download it with:

    ```bash
    python cached_fineweb10B.py
    ```

    Files are saved under `fineweb10B/` as `fineweb_train_*.bin` and `fineweb_val_*.bin`.

2.  **Run full training.**

    The training script supports multi-device training (via `jax.pmap`) and automatic gradient accumulation to reach a target global token batch size. It now runs with built-in defaults (no CLI args).

    Run:

    ```bash
    python3 train_gpt.py
    ```

    Defaults are defined in `train_gpt.py` under the `Hyperparameters` dataclass. To change them, edit that class (e.g., `batch_size`, `sequence_length`, `total_batch_size`, `learning_rate`, etc.).

    Notes:

    - tokens per fwd/bwd: `batch_size * sequence_length * num_devices`
    - grad accumulation steps (Ng): `total_batch_size / tokens_per_fwd_bwd`
    - dtype can be set via `Hyperparameters.dtype` ("float32" or "bfloat16").

    The script logs train loss each step and validation loss every `val_loss_every` steps.

## Training Speed History

Hardware: 8x H100 (fixed). All entries use the same training config.
| # | Record time | Description | Date |
| - | - | - | - |
| 1 | 176 min | Initial baseline | 2025-09-06 |
