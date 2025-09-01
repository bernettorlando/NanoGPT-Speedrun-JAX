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

    The training script uses the FineWeb 10B dataset. You can download it by running:

    ```bash
    python cached_fineweb10B.py
    ```

    This will download the dataset into a `fineweb10B` directory.

2.  **Run the training script.**

    Once the dataset is downloaded, you can start training the model:

    ```bash
    python train_gpt.py
    ```

    The script is configured to train a 124M parameter model and will print the loss every 100 steps. You can modify the hyperparameters in `train_gpt.py` to suit your needs.
