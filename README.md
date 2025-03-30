# Mixed Curvature Transformers

This project explores the geometric properties of representation spaces in transformer models, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Motivation

Recent optimizers like [Muon](https://github.com/KellerJordan/Muon) achieve remarkable efficiency by orthogonalizing weights in neural networks. This orthogonalization creates more spread singular value decompositions (SVDs), which appears to improve network training dynamics.

The success of these approaches suggests that representation spaces in transformers may be inherently anisotropic - not uniformly distributed in all directions. This anisotropy could be a fundamental property that limits the effectiveness of standard Euclidean optimization approaches.

Research such as [Hyperbolic Geometric Latent Diffusion Model for Graph Generation](https://arxiv.org/abs/2405.03188) has demonstrated that using hyperbolic geometry can better capture the underlying structure of certain data types by accounting for this anisotropy.

## Project Goal

This project aims to implement and test mixed curvature transformers - models that combine Euclidean, hyperbolic, and potentially other geometries to better match the intrinsic structure of the representation space. By adapting the geometric properties of the model to the natural geometry of the data, we hypothesize improved training efficiency and overall performance.

## Implementation

We use the minimal and efficient nanoGPT implementation as our foundation, modifying key components to incorporate non-Euclidean geometries. The code is designed to be as lightweight and readable as possible while enabling meaningful experiments in representation geometry.

## Dataset Preparation

For experimenting with mixed curvature transformers, we use the same dataset preparation approach as the original nanoGPT:

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

Both datasets are prepared to be used with the training scripts. For mixed curvature experiments, we can use these datasets to compare performance against baseline Euclidean transformer architectures.

## Getting Started

[Installation and usage instructions will be added as the project develops]

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Geometric optimization insights from [Muon](https://github.com/KellerJordan/Muon) by Keller Jordan et al.



