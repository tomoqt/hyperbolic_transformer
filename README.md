

# Transformer Block Looping Experiment

This project explores iterating transformer blocks multiple times within a single forward pass, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Motivation

Standard transformer models process input through a fixed sequence of layers. This project investigates an alternative approach where specific blocks or ranges of blocks within the transformer architecture can be applied iteratively. The core idea is to explore whether repeatedly applying the transformations of certain layers can lead to deeper processing, potentially improving model performance or efficiency under certain conditions.

This concept is related to recurrent mechanisms and adaptive computation, where the model might dynamically adjust the amount of processing applied based on the input or intermediate representations.

## Project Goal

This project aims to implement and evaluate transformers capable of looping specific blocks. Key aspects include:
*   **Flexible Looping:** Allowing configuration of which block(s) to loop (`loop_center_idx`, `loop_radius`).
*   **Looping Strategies:** Implementing both "tied" looping (iterating the entire range as one unit) and "untied" looping (iterating each block within the range independently) (`tied_looping`).
*   **Residual Connections:** Exploring different ways to handle the residual stream during loops, such as standard addition or concatenating the initial pre-loop representation and adapting it (`concatenate_initial_representation`).
*   **Noise Injection:** Optionally adding noise during the first loop iteration (`loop_noise_scale`).
*   **Adaptive Computation:** Implementing an automatic exit mechanism to stop looping based on representation convergence (`automatic_loop_exit`, `automatic_loop_exit_threshold`).
*   **Analysis:** Providing tools to track representations across loops for analysis (`loops_representation`).

The goal is to understand the impact of these looping mechanisms on training dynamics and model capabilities compared to standard transformer architectures.

## Implementation

We use the minimal and efficient nanoGPT implementation as our foundation, modifying the `GPT` model in `model.py` to incorporate the looping functionality described above. The core changes are within the `forward` method of the `GPT` class and the corresponding additions to the `GPTConfig` dataclass.

## Dataset Preparation

For experimenting with transformer block looping, we use the same dataset preparation approach as the original nanoGPT:

### Shakespeare Dataset (Small Scale Testing)

For quick experimentation, the Shakespeare dataset provides a lightweight option:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin` files with character-level tokenization.

### OpenWebText Dataset (Full Scale Training)

For more extensive training, prepare the OpenWebText dataset:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the OpenWebText dataset, creating `train.bin` and `val.bin` files with GPT-2 BPE tokenization.

Both datasets are prepared to be used with the training scripts. For transformer block looping experiments, we can use these datasets to compare performance against baseline transformer architectures.

## Getting Started

[Installation and usage instructions will be added as the project develops]

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy



