Consulting paper on Denoising Diffusion Probabilistic Models by Jonathan Ho, Ajay Jain, Pieter Abbel

# Diffusion Model from Scratch (DDPM)

This repository contains a clean, modular implementation of a Denoising Diffusion Probabilistic Model (DDPM) built using PyTorch. The project is designed to understand the core mechanics of generative diffusion models, including the forward diffusion process (adding noise) and the reverse process (denoising/generation) using a conditioned U-Net architecture.

# Project Overview

The goal of this project is to implement a complete, functional diffusion model, focusing on clarity and modularity. This model is capable of generating high-quality images conditioned on class labels, following the principles laid out in the original DDPM paper, "Denoising Diffusion Probabilistic Models."

### Key Features

Custom Scheduler: Implements a cosine noise schedule for stable and high-quality diffusion steps.

Modular U-Net: Features a U-Net backbone with integrated Transformer Positional Embeddings for timestep conditioning and Class Embeddings for conditional generation.

Attention Mechanisms: Includes self-attention blocks within the U-Net at lower resolutions to capture long-range dependencies in the feature maps.

Full Pipeline: Contains functions for the forward diffusion process, reverse sampling, and training loop components.

### Architecture

The model is divided into three primary Python files for maximum clarity and organization:

unetModel.py

The core neural network responsible for predicting the noise ($\epsilon$) added at each timestep. This is a modified U-Net that includes:

Timestep Conditioning: The diffusion timestep ($t$) is encoded using TransformerPositionalEmbedding and injected into every ResNetBlock.

Class Conditioning: Includes a nn.Embedding layer to allow for class-conditional image generation (e.g., generating a specific type of image like "cat" or "car").

Downsample/Upsample Blocks: Uses standard convolutional blocks with Group Normalization and SiLU activation.

layers.py

Contains all the reusable building blocks for the U-Net:

TransformerPositionalEmbedding: Sinusoidal embedding layer for timesteps.

ResNetBlock: Wide ResNet block with timestep embedding injection.

ConvBlock, AttentionBlock: Standard utility modules for convolution and self-attention.

ConvDownBlock, ConvUpBlock, AttentionDownBlock, AttentionUpBlock: Combination blocks used to structure the encoder and decoder paths of the U-Net.

myDiffusionModel.py

This file handles the high-level diffusion process logic:

cosine_beta_schedule: Generates the variance schedule ($\beta_t$) and pre-calculates the necessary $\alpha$ terms ($\alpha_t$ and $\bar{\alpha}_t$).

forward_diffusion: Adds noise to a clean image ($x_0$) to produce a noisy image ($x_t$) at a given timestep ($t$).

reverse_sampling: Implements the iterative reverse process to sample a clean image from pure noise.

Training/Testing Functions: Includes the primary data loading, model instantiation, and visualization logic.

## Setup and Installation

Prerequisites

You must have Python installed (preferably Python 3.9+).

Installation Steps

Clone the repository:

git clone [Your Repository URL Here]
cd diffusion-model-scratch


Install dependencies:
The project relies on PyTorch, NumPy, and Matplotlib. All required packages are listed in requirements.txt.

pip install -r requirements.txt


🏃 Usage

1. Training the Model

(Note: The current repository structure focuses on implementation. You will need to add a dedicated training loop function that calls the forward pass, calculates the loss (L2 norm between predicted noise and true noise), and performs backpropagation.)

Placeholder for future training script:

python train.py


2. Generating Images

To demonstrate the reverse sampling process, you can run the main logic within myDiffusionModel.py. The provided snippet in this file shows how to set up the model and perform the sampling process:

# From myDiffusionModel.py
if __name__ == "__main__":
    # ... setup code ...
    generated_images = reverse_sampling(
        # ... arguments for the UNet model, schedule parameters, etc.
    )
    # ... visualization code ...


To execute the current main file:

python myDiffusionModel.py


This script will initialize the U-Net, run the reverse diffusion process, and display a set of intermediate generated images using Matplotlib.