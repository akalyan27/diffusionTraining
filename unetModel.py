
import torch
import torch.nn as nn

from layers import ConvDownBlock, \
    AttentionDownBlock, \
    AttentionUpBlock, \
    TransformerPositionalEmbedding, \
    ConvUpBlock

class UNet(torch.nn.Module):
    '''
    Modified UNet architecture as described in the DDPM paper and implemented in 
    https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/unet.py
    '''

    def __init__(self, image_size=256, input_channels=3):
        '''
        Group normalization is used instead of weight normalization. 
        For 32×32 inputs, the model uses four resolution levels (from 32×32 down to 4×4); 
        for 256×256 inputs, six levels are typical (this implementation uses five).
        Each resolution level contains two residual convolutional blocks, with self-attention applied 
        at the 16×16 resolution between the conv blocks (Ref: Self-Attention paper).
        The diffusion timestep t is embedded into each residual block using sinusoidal position 
        embeddings from the Transformer architecture (Ref: DDPM paper)
        '''
        super().__init__()

        self.class_embed = nn.Embedding(10, 128) #10 labels and 128 dimensions 
        self.class_embed.to('cpu')  # Has to be in CPU for function calls in forward

        self.class_project = nn.Sequential(
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=256, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4)
        ])

        self.bottleneck = AttentionDownBlock(in_channels=512, out_channels=512, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128*4, downsample=False)                                                                                                  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(in_channels=512 + 512, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            AttentionUpBlock(in_channels=512 + 256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=128 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=256, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, padding=1)
        )

    def forward(self, input_tensor, time, class_label):

        time_encoded = self.positional_encoding(time)

        class_label = class_label.cpu().long()

        class_emb = self.class_embed(class_label)
        class_emb = self.class_project(class_emb.to(time_encoded.device))        

        cond_emb = time_encoded + class_emb

        initial_x = self.initial_conv(input_tensor)

        states_for_skip_connections = [initial_x]

        x = initial_x
        
        for i, block in enumerate(self.downsample_blocks):
            x = block(x, time_encoded)
            states_for_skip_connections.append(x)
        states_for_skip_connections = list(reversed(states_for_skip_connections))

        x = self.bottleneck(x, time_encoded)

        for i, (block, skip) in enumerate(zip(self.upsample_blocks, states_for_skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)

        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        # Get initial shape [3, 256, 256] with convolutions
        out = self.output_conv(x)

        return out

        return out


