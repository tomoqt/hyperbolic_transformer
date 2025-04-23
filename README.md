# Mixed Curvature Transformers

This project explores the geometric properties of representation spaces in transformer models, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## (initial) Motivation

Recent optimizers like [Muon](https://github.com/KellerJordan/Muon) achieve remarkable efficiency by orthogonalizing weights in neural networks. This orthogonalization creates more spread singular value decompositions (SVDs), which appears to improve network training dynamics.

The success of these approaches suggests that representation spaces in transformers may be inherently anisotropic - not uniformly distributed in all directions. This anisotropy could be a fundamental property that limits the effectiveness of standard Euclidean optimization approaches.

Research such as [Hyperbolic Geometric Latent Diffusion Model for Graph Generation](https://arxiv.org/abs/2405.03188) has demonstrated that using hyperbolic geometry can better capture the underlying structure of certain data types by accounting for this anisotropy.

As the project carried on, we noticed a significant speedup of the model's training under some circumstances. Specifically, we ran experiments on gpt-2 variants. One on the shakespeare-char dataset (a small, test dataset tokenized char-whise) and on fineweb 10b, on an 85M model. 

This project contains a few key architectural choices that seemingly allow this:
- the model uses mobius addition to define sum in constant, curved space. Consequently, it introduces the exponential map and its inverse. The idea is to map representations to constantly curved space, while handling the regular neural network operations in the tangent space, which is locally euclidean. To do so, the model projects inputs back to tangent space right before attention, and projects embeddings back to curved space right after.
- the curvature parameter, defining mobius addition and hence the rest of the geometry, is learnable. This is also explore in works like https://arxiv.org/pdf/2309.04082 , which makes this effectively a type of stereographic model, which (so far) only extends to negative curvature. We specifically make this parameter learnale block-wise and head-wise (but we can also tie this in both directions).
- Importantly, the reference point to operate the mapping to the tangent space, for each token, is represented by the embedding of the previous token. This doeesn't break causality and empirically seems to work pretty well. Still have to figure out exacly why it works better than aggregating positions between tokens (as done for GNNs, could probably be extended to do some kind of aggregation over a past window of tokens tho.


## Interesting results:

speedup ![image](https://github.com/user-attachments/assets/c73726ab-0a09-4155-a2da-96097d56ce2f)


Specifically, interestingly not projecting embedding table to curved space seems to work better (but we haven't tried all combinations on this, i have hundreds of different experiemnts but not well documented)
We can log curvature, interestingly, block 0 usuall seems to have increasing curvature, while the rest decreases more or less sharply. 


![image](https://github.com/user-attachments/assets/eb8636ee-a4f3-4c70-9825-5dfbe28e2067)

Unfortunately i haven't really logged these properly, so i can't really systematically give you an idea about these. But cool to look at. 


## Anisotpy vs isotropy

Ok, elephant in the room. Effect seems to be there for at least the shakespeare-char. On fineweb, it looks more like a mixed bag. You can find some scripts in the repo with results about this. I attach some examples:

shakespeare-char: 

![image](https://github.com/user-attachments/assets/7a41ef52-fc9e-424f-b2b4-cbd08ea7700d)

fineweb:

![image](https://github.com/user-attachments/assets/39a5514b-a4ab-4fbe-8cdc-e197dd4ad0c8)


## Clamping and instability
Since stereographic operations require heavy use of hyperbolic functions, the model is pretty unstable and prone to nans. We used quite a few clamps / epsilons in the code to relieve some of the division by small number problems. there probably is a better way to smoothen this out. 

## Project Goal

This project aims to implement and test mixed curvature transformers - models that combine Euclidean, hyperbolic, and potentially other geometries to better match the intrinsic structure of the representation space. By adapting the geometric properties of the model to the natural geometry of the data, we hypothesize improved training efficiency and overall performance.

## Current objectives and nice to haves:
This work does go in the direction of learning *more* in the transformer architecture. I think this itself subtracts structure and may allow for more expressive models overall. In order to try and show this, i believe we should:
- focus on doing a sweep to try and get a record in the speedrun. we had several attempts with prety bad failures, mainly due to gradients exploding, irrespective of lr (swept). so it's likely a very fragile setup for us.
- focus a little on writing some of the underlying theory in a paper.
- improve the architecture by adding:
    - positive curvature
    - more stable ops
    - different reference points calculation methods
    - possibily fully stereographic ops

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

Fineweb
```sh
python data/fineweb/prepare.py
```

This downloads and tokenizes the Fineweb10B dataset, creating `train.bin` and `val.bin` files with GPT-2 BPE tokenization.

Both datasets are prepared to be used with the training scripts. For mixed curvature experiments, we can use these datasets to compare performance against baseline Euclidean transformer architectures.

## Getting Started

[Installation and usage instructions will be added as the project develops]

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Geometric optimization insights from [Muon](https://github.com/KellerJordan/Muon) by Keller Jordan et al.



