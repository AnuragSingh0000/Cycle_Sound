import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer that maps image tokens to audio token logits.
class ImagetoAudioTransformer(nn.Module):
    def __init__(self, latent_dim=512, nhead=8, num_layers=4, 
                audio_token_len=64, image_token_len=64, vocab_size=512):
        super(ImagetoAudioTransformer, self).__init__()
        self.audio_token_len = audio_token_len  # expected length of audio tokens (after interpolation if needed)
        self.image_token_len = image_token_len  # e.g., 8x8=64 tokens for a 64-token image representation
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        # Positional embeddings for audio tokens
        self.audio_pos_emb = nn.Parameter(torch.randn(1, audio_token_len, latent_dim))
        # Positional embeddings for image tokens (target sequence)
        self.image_pos_emb = nn.Parameter(torch.randn(1, image_token_len, latent_dim))
        
        self.image_proj = nn.Linear(latent_dim, latent_dim)
        
        # Transformer decoder to predict audio tokens
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection to logits over the codebook vocabulary
        self.token_logits = nn.Linear(latent_dim, vocab_size)
    
    def forward(self, image_tokens):
        B = image_tokens.size(0)
        # Ensure image_tokens have fixed length (interpolate if necessary)
        if image_tokens.size(1) != self.image_token_len:
            image_tokens = F.interpolate(image_tokens.transpose(1,2), 
                                        size=self.image_token_len, mode='linear', align_corners=False)
            image_tokens = image_tokens.transpose(1,2)
        # Project tokens and add positional embeddings.
        image_emb = self.image_proj(image_tokens) + self.image_pos_emb  # [B, image_token_len, latent_dim]
        
        # For inference, initialize audio embeddings with learned audio positional embeddings.
        gen_audio_emb = self.audio_pos_emb.repeat(B, 1, 1)
    
        # Transformer expects [sequence, batch, features]
        gen_audio_emb = gen_audio_emb.transpose(0, 1)      # [audio_token_len, B, latent_dim]
        image_emb = image_emb.transpose(0, 1)              # [image_token_len, B, latent_dim]
        
        # Create causal mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(gen_audio_emb.size(0)).to(gen_audio_emb.device)
        transformer_output = self.transformer_decoder(gen_audio_emb, image_emb, tgt_mask=tgt_mask)
        logits = self.token_logits(transformer_output.transpose(0, 1))  # [B, audio_token_len, vocab_size]
        return logits

# Combined model that connects the image encoder and the transformer.
class ImagetoAudio(nn.Module):
    def __init__(self, image_encoder, image_to_audio_transformer):
        super(ImagetoAudio, self).__init__()
        self.image_encoder = image_encoder
        self.image_to_audio_transformer = image_to_audio_transformer
        
    def forward(self, image):
        image_features = self.image_encoder(image)  # [B, tokens, latent_dim]
        logits = self.image_to_audio_transformer(image_features)
        return logits
