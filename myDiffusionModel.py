#design scheduler/ method to generate beta values (noise variance added at each step of forward diffusion)

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

from unetModel import UNet 
from torchvision import datasets
from torch.utils.data import DataLoader


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
# Given a noisy image, predict the noise (w neural network) and then use that to denoise the image

def reverse_diffusion(xt, t, myModel, class_label, alphas, alphas_cumprod, betas):
    '''
    denoise the image to get image at x(t-1) using x(t) and predicted noise
    eq = Mean + Standard Deviation * Random Constant
    x(t-1) = 1/sqrt(alphas_cumprod(t))  *  (x(t) - sqrt(1 - alphas_cumprod(t)) * neuralNetwork_predicted_noise) + 
                standard_deviation * random_constant
    '''

    #The real training and inference happens here at predicting noise
    predicted_noise = myModel(xt, t, class_label) 
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    betas_t = betas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)

    # Calculate the standard deviation and random constant to add to mean
    sigma = torch.sqrt(betas_t)
    z = torch.randn_like(xt)

    # mean + sigma * z
    eq = (1.0/ torch.sqrt(alpha_bar_t)) * (xt - torch.sqrt(1 - alpha_t)) * predicted_noise + (sigma * z)

    return eq

'''
 Actual Model to predict noise - UNet becuase it is fully convolutional 
    - retains all spatial dimensions as opposed to a CNN which flattens data and is used for classification)
    - good for images
    - have to modify the UNet architecture to accomodate for timestep embeddings
 implemented in unetMode.py 
'''

def generate_from_noise(model, num_diffusion_steps, alphas, alphas_cumprod, betas, class_label, device, image_shape=(3,32,32), show_every=50):
    model.eval()
    with torch.no_grad():
        img = torch.randn((1, *image_shape)).to(device)  # Start from pure noise
        generated_images = []

        # if class_label is not None:
        #     label = torch.tensor([class_label], device=device)

        label = class_label.to("cpu") # UNet class embedding layer is on CPU

        # has to go through all diffusion steps in order to denoise properly. 
        for t in reversed(range(num_diffusion_steps)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)  # Create a batch of the current timestep

            predicted_noise = model(img, t_batch, label) 

            class_num = class_label.item() 

            img = reverse_diffusion(img, t_batch, model, alphas, alphas_cumprod, betas, class_label)
            img = torch.clamp(img, -1.0, 1.0)  # Ensure the image stays within valid range

            if t % show_every == 0 or t == num_diffusion_steps - 1 or t == 0:
                print(f"Step {t}/{num_diffusion_steps}")
                img_clip = img.clamp(-1, 1)
                img_np = ((img_clip + 1) / 2).cpu().permute(1,2,3,0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                generated_images.append((num_diffusion_steps - t, img_np))
    return generated_images





if __name__ == "__main__":

    #use pytorch's given transformer architecture - eventual step to create my own
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
    model = UNet(image_size=32, input_channels=3).to(device)
    model.class_embed.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # tr_photo = "content/happy-smile.png"
    # image = Image.open(tr_photo)
    # print(f"Using device: {device}")
    
    IMAGE_SHAPE = (32,32)
    num_epochs = 10

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # x0 = transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension
    # print(f"Original image (x0) shape after transform: {x0.shape}")

    # Get all ingredients for the forward process shown as sampling process in the paper
    # x(t) = sqrt(alphas_cumprod(t)) * x0 + sqrt(1 - alphas_cumprod(t)) * NOISE
    num_diffusion_steps = 1000
    beta_values = cosine_beta_schedule(num_diffusion_steps).to(device) 
    alphas = 1. - beta_values
    alphas_cumprod = torch.cumprod(alphas, dim=0) #cumulative product of alphas
    ## alphas_cumprod is used to scale the original image at each step

    sqrt_alphas_scaling = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_scaling = torch.sqrt(1 - alphas_cumprod).to(device)

    #TRAINING PROCESS - predict the noise given a noised image at time t and analyze with MSE loss
    # for epoch in range(num_epochs):
    #     mean_epoch_loss = []

    #     for batch, labels in dataloader:
    #         batch = batch.to(device)
    #         #Random timesteps for each sample in batch
    #         time = torch.randint(0,num_diffusion_steps, (batch.size(0),), device=device)

    #         xt_noisey_image, noise = forward_diffusion(
    #             batch,
    #             time,
    #             sqrt_alphas_scaling,
    #             sqrt_one_minus_alphas_scaling
    #         )

    #         predicted_noise = model(xt_noisey_image, time, labels) 

    #         optimizer.zero_grad()
    #         loss = f.mse_loss(predicted_noise, noise)
    #         loss.backward()
    #         optimizer.step()
            
    #         mean_epoch_loss.append(loss.item())

    #     # Was previously printing the noisy image in every iteration but now just printing one at end of epoch to help visualize the training step
    #     noise_image_noBatch = xt_noisey_image[0].squeeze(0)  # Remove batch dimension (only used for transforms expecting batch size)
    #     noise_image = reverse_transform(noise_image_noBatch)
    #     output_image = "noiseImage_" + str(epoch) + ".png"
    #     output_folder = "/Users/adithyakalyan/Projects/games/content/myImages"
    #     os.makedirs(output_folder, exist_ok=True)
    #     output_path = os.path.join(output_folder, output_image)
    #     noise_image.save(output_path)

    #     mean_epoch_loss = np.mean(mean_epoch_loss)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_epoch_loss:.4f}")

    # Reverse generate the images and show our progress as we step through 
    CIFAR_label_map = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    print("CIFAR-10 classes:", CIFAR_label_map)
    requested_class = str(input("Enter a class label to generate an image: "))
    class_num = 0
    try: 
        class_num = int(CIFAR_label_map[requested_class])
    except ValueError:
        print("Invalid class label. Please enter one of the possible classes as stated above. Reverted to default airplane")
        requested_class = "airplane"
        class_num = 0

    if requested_class not in CIFAR_label_map:
        print("Invalid class label. Please enter one of the possible classes as stated above. Reverted to default airplane")
        class_num = 0
        requested_class = "airplane"

    print(f"Generating image based on class: {requested_class}")   

    class_num = torch.tensor([class_num], dtype=torch.long, device=device)
    print(f"label and type: {class_num}, {class_num.dtype}")

    
    generated_images = generate_from_noise(
        model,
        num_diffusion_steps,
        alphas,
        alphas_cumprod,
        beta_values,
        class_num,
        device,
        image_shape=(3,32,32),
        show_every=200
    )

    #prints generated images side by side
    fig, axes = plt.subplots(1, len(generated_images), figsize = (20,5))
    if len(generated_images) == 1:
        axes = [axes]

    for ax, (steps, img_np) in zip(axes, generated_images):
        img_np = img_np.squeeze()  # Remove batch dimension 
        img_np = np.transpose(img_np, (1, 2, 0))
        ax.imshow(img_np)
        ax.set_title(f"Step {steps}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    #actual forward diffusion process and testing imtermediately
    # timesteps_to_test = [0, 50, 100, 200, 500, 999]

    # for t in timesteps_to_test:

    #     xt_noisey_image, noise = forward_diffusion(
    #         x0,
    #         t,
    #         sqrt_alphas_scaling,
    #         sqrt_one_minus_alphas_scaling
    #     )
    #     xt_noisey_image_noBatch = xt_noisey_image.squeeze(0)  # Remove batch dimension (only used for transforms expecting batch size)
    #     noise_image = reverse_transform(xt_noisey_image_noBatch)

    #     output_image = "noiseImage_" + str(t) + ".png"
    #     output_folder = "/Users/adithyakalyan/Projects/games/content/myImages"
    #     os.makedirs(output_folder, exist_ok=True)
    #     output_path = os.path.join(output_folder, output_image)
    #     noise_image.save(output_path)

