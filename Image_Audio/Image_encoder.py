import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.nn import ReLU, LeakyReLU, Tanh
from einops import einsum, rearrange

device = torch.device("cuda")

class Encoder(nn.Module):
    def __init__(self, num_embeddings):
        super(Encoder, self).__init__()
        # Encoder consists of Conv blocks
        self.encoder_layers = nn.ModuleList([
            # input size, output size, filter, stride, padding
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU()
        ])
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU())
        ])
        # Conv for outputting an embedding so that it can be matched with codebook
        self.encoder_quant_conv = nn.Sequential(
            nn.Conv2d(64, num_embeddings, 1)
        )
        
    def forward(self, x):
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
        for layer in self.residuals:
            out = out + layer(out)
        out = self.encoder_quant_conv(out)
        return out
    
class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Quantizer, self).__init__()
        
        self.num_embeddings = num_embeddings  # Total number of discrete embeddings
        self.embedding = nn.Embedding(self.num_embeddings, embedding_dim).to(device) # Codebook (Embedding matrix)
        
    def forward(self, x):
        B, C, H, W = x.shape  # Extract batch size, channels, height, width
        x = x.to(device)
        
        # Apply Gumbel-Softmax to convert logits into a probability distribution over discrete embeddings
        one_hot = torch.nn.functional.gumbel_softmax(x, tau=0.9, dim=1, hard=False)
        
        # Compute the weighted sum of the embeddings based on the one-hot probabilities
        sampled = einsum(one_hot, self.embedding.weight, 'b n h w, n d -> b d h w')
        
        # Compute KL Divergence to encourage entropy maximization
        logits = rearrange(x, 'b n h w -> b (h w) n')  # Reshape for better computation
        log_qy = torch.nn.functional.log_softmax(logits, dim=-1)  # Compute log softmax probabilities
        
        # Compute KL divergence with a uniform prior over embeddings
        log_uniform = torch.log(torch.tensor([1. / self.num_embeddings], device=x.device))
        kl_div = torch.nn.functional.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
        
        return sampled, kl_div, logits, log_qy
    
    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')


class ImageEncoder(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        self.encoder = Encoder(num_embeddings).to(device)
        self.quantizer = Quantizer(num_embeddings, embedding_dim).to(device)
    
    def get_codebook_indices(self, x):
        x = x.to(device)
        enc_logits = self.encoder(x)  # Get logits from encoder
        indices = torch.argmax(enc_logits, dim=1)  # Get indices of max logit (hard quantization)
        return indices
    
    def forward(self, x):
        x = x.to(device)
        enc = self.encoder(x)  # Encode input image
        quant_output, kl, logits, log_qy = self.quantizer(enc)  # Quantize representation
        B, C, H, W = quant_output.shape
        quant_output = quant_output.permute(0, 2, 3, 1)  # [B, H, W, C]
        quant_output = quant_output.reshape(B, H * W, C)  # [B, H*W, C]

        return quant_output
