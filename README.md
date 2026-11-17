# CS-Assignment-6-Vision-Transformer
CS Assignment #6 Vision Transformer
# Lab 6 — Patch Size in a Small Vision Transformer (CIFAR-10)

## Contents
- `cv6_vit_patch_size.ipynb` — executed notebook (data split, ViT, shape trace, attention example, both training runs, comparison)
- `analysis.pdf` — 400–600 word analysis
- `outputs/split_indices.npz` — saved train/validation indices into the official CIFAR-10 training split (seed 675)
- `outputs/config.json` — full configuration, normalization statistics, library versions and hardware
- `outputs/results.csv`, `outputs/histories.json` — results table and per-epoch curves
- `outputs/*.png` — figures

## Setup and run 
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install numpy matplotlib pandas jupyter
jupyter nbconvert --to notebook --execute --inplace cv6_vit_patch_size.ipynb
```
CIFAR-10 is downloaded automatically to `data/` by torchvision (not included in this ZIP). The official test split
is never loaded. Measured training runtime on a 2-core GitHub Codespaces CPU: 86.7 s (4×4) and 19.1 s (8×8); the CIFAR-10 download can add 10–20 minutes.

## Configuration summary
Stratified 10,000 train / 2,000 validation images from the official training split (1,000 / 200 per class), seed 675,
per-channel normalization computed on the training subset, no augmentation. ViT: P×P non-overlapping patches,
shared linear projection to 64, learnable class token and positional embeddings, 2 pre-norm encoder blocks,
4 heads, MLP 128, dropout 0.1, linear head on the class token. Adam, lr 3e-4, batch 64, 5 epochs, cross-entropy on logits.
The full ("not reduced") compute setting was used.

## Sources
PyTorch `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` documentation; Dosovitskiy et al., ICLR 2021;
CIFAR-10 (Krizhevsky, 2009).
