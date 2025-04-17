import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioEncoder(nn.Module):
    def __init__(self, num_embeddings):
        super(AudioEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # [B, 32, 80, 88]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # [B, 64, 80, 88]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # [B, 64, 40, 44]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # [B, 128, 40, 44]
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # [B, 128, 20, 22]
            nn.ReLU(),
        )
        self.encoder_quant_conv = nn.Conv2d(128, num_embeddings, kernel_size=1)

    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.encoder_quant_conv(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.5, 0.5)
    
    def forward(self, x):
        B, C, H, W = x.shape
        one_hot = F.gumbel_softmax(x, tau=0.9, hard=True, dim=1)
        quantized = torch.einsum('b n h w, n d -> b d h w', one_hot, self.embedding.weight)
        
        logits = rearrange(x, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        uniform_target = torch.full_like(log_qy, 1.0 / self.num_embeddings)
        kl_div = F.kl_div(log_qy, uniform_target, reduction='batchmean', log_target=False)
        return quantized, self.commitment_cost * kl_div, one_hot

    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')
        
class AudioDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super(AudioDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 3, stride=1, padding=1),  # [B, 128, 20, 22]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),  # [B, 128, 40, 44]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # [B, 64, 40, 44]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),  # [B, 64, 80, 88]
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),  # [B, 1, 80, 88]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)

class AudioAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(AudioAE, self).__init__()
        self.encoder = AudioEncoder(num_embeddings)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = AudioDecoder(embedding_dim)
    
    def forward(self, x):
        logits = self.encoder(x)
        quantized, loss, one_hot = self.quantizer(logits)
        recon = self.decoder(quantized)
        return recon
    
    def get_codebook_indices(self, x):
        logits = self.encoder(x)
        one_hot = F.gumbel_softmax(logits, tau=0.9, hard=True, dim=1)
        tokens = one_hot.argmax(dim=1)
        return tokens
    
    def decode_from_logits(self, logits):
        B, audio_token_len, vocab_size = logits.shape
        predicted_indices = torch.argmax(logits, dim=-1)  # shape: [B, audio_token_len]
        predicted_indices_grid = predicted_indices.view(B, 20, 22)
        
        # Convert indices to one-hot: output shape [B, vocab_size, side, side]
        indices = F.one_hot(predicted_indices_grid, num_classes=vocab_size).permute(0, 3, 1, 2).float()
        indices = indices.to(device)
        quantized_indices = self.quantizer.quantize_indices(indices)  # Convert indices back to embeddings
        return self.decoder(quantized_indices)  # Decode into a spect
    