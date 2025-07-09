#design scheduler/ method to generate beta values (noise variance added at each step of forward diffusion)

import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


def cosine_beta_schedule(num_diffusion_steps, factor=0.008):
    """
    Generates a cosine beta scheduler for the diffusion process. 
    The beta values are used to control the noise variance added at each step of the forward diffusion process.
    Args:
        num_diffusion_steps (int): Number of diffusion steps.
        factor (float): small constant used in cosine calculations 
    """
    steps = torch.arange(0, num_diffusion_steps + 1, 1, dtype=torch.float32)
    prod = torch.cos((steps / num_diffusion_steps + factor) * torch.pi * 0.5)
    prod = prod ** 2
    alpha_bar_t = prod / prod[0]  # Normalize to ensure alpha_bar_0 = 1.0
    
    # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    # essentially shifted one time step back and adds a 1 to the front [1.0,1.0, alpha_bar_1, ..., alpha_bar_t-1]
    alpha_bar_t_minus_1 = torch.cat([torch.tensor([1.0]), alpha_bar_t[:-1]]) 
    betas = 1 - (alpha_bar_t / alpha_bar_t_minus_1)
   
    return torch.clip(betas, 0.0001, 0.9999)
    # print(steps)

def forward_diffusion(x0, t, sqrt_alphas_scaling, sqrt_one_minus_alphas_scaling):

    '''
      Forward diffusion process where we have to take the equation and apply
      it to the input image(tensor) at time t
    '''
    #.view(-1,1,1,1) essentially infers the size of that dimension and reshapes your data so it 
    # can be broadcasted correctly across the tensors dimensions
    noise = torch.randn_like(x0) # This is the epsilon in the formula
    sqrt_alphas_scaling = sqrt_alphas_scaling[t].view(-1, 1, 1, 1)  
    sqrt_one_minus_alphas_scaling = sqrt_one_minus_alphas_scaling[t].view(-1, 1, 1, 1)


    xt = sqrt_alphas_scaling * x0 + sqrt_one_minus_alphas_scaling * noise
    
    return xt, noise

## REVERSE DIFFUSION PROCESSSS



if __name__ == "__main__":


    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])

    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert to NumPy array
        transforms.ToPILImage(),
    ])

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr_photo = "content/happy-smile.png"
    image = Image.open(tr_photo)
    print(f"Using device: {device}")
    
    IMAGE_SHAPE = (32,32)

    x0 = transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension
    print(f"Original image (x0) shape after transform: {x0.shape}")



    # Get all ingredients for the forward process shown as sampling process in the paper
    # x(t) = sqrt(alphas_cumprod(t)) * x0 + sqrt(1 - alphas_cumprod(t)) * NOISE
    num_diffusion_steps = 1000
    beta_values = cosine_beta_schedule(num_diffusion_steps)
    alphas = 1. - beta_values
    alphas_cumprod = torch.cumprod(alphas, dim=0) #cumulative product of alphas
    ## alphas_cumprod is used to scale the original image at each step

    sqrt_alphas_scaling = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_scaling = torch.sqrt(1 - alphas_cumprod).to(device)

    #actual forward diffusion process and testing imtermediately
    timesteps_to_test = [0, 50, 100, 200, 500, 999]

    for t in timesteps_to_test:

        xt_noisey_image, noise = forward_diffusion(
            x0,
            t,
            sqrt_alphas_scaling,
            sqrt_one_minus_alphas_scaling
        )

        xt_noisey_image_noBatch = xt_noisey_image.squeeze(0)  # Remove batch dimension (only used for transforms expecting batch size)

        noise_image = reverse_transform(xt_noisey_image_noBatch)

        output_image = "noiseImage_" + str(t) + ".png"
        output_folder = "/Users/adithyakalyan/Projects/games/content/myImages"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_image)
        noise_image.save(output_path)

