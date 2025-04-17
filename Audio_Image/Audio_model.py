import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Audio-to-Image Generation Model
#############################################

# Audio Encoder: converts spectrogram to a sequence of latent tokens.
class AudioEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, latent_dim=256):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4 = nn.BatchNorm2d(latent_dim)
        
    def forward(self, x):
        # Input x: [B, 1, H, W] spectrogram
        x = F.relu(self.bn1(self.conv1(x)))  # [B, base_channels, H/2, W/2]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, base_channels*2, H/4, W/4]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, base_channels*4, H/8, W/8]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, latent_dim, H/16, W/16]
        B, C, H, W = x.shape
        # Flatten spatial dimensions as tokens for the transformer
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, tokens_audio, latent_dim]
        return x

# Transformer that maps audio tokens to image token logits.
class AudioToTokenTransformer(nn.Module):
    def __init__(self, latent_dim=256, nhead=8, num_layers=4, 
                audio_token_len=64, image_token_len=256, vocab_size=1024):
        super(AudioToTokenTransformer, self).__init__()
        self.audio_token_len = audio_token_len
        self.image_token_len = image_token_len
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        # Positional embeddings for audio tokens
        self.audio_pos_emb = nn.Parameter(torch.randn(1, audio_token_len, latent_dim))
        # Positional embeddings for image tokens (target sequence)
        self.image_pos_emb = nn.Parameter(torch.randn(1, image_token_len, latent_dim))
        
        # Linear projection for audio features
        self.audio_proj = nn.Linear(latent_dim, latent_dim)
        
        # Transformer decoder to predict image tokens
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Token embedding for ground-truth image tokens (learned embedding for each codebook index)
        self.token_embedding = nn.Embedding(vocab_size, latent_dim)
        # Final projection to logits over the codebook vocabulary
        self.token_logits = nn.Linear(latent_dim, vocab_size)
    
    def forward(self, audio_tokens):
        """
        Args:
            audio_tokens: [B, tokens_audio, latent_dim] from the audio encoder.
            target_tokens: [B, image_token_len] ground truth tokens.
        Returns:
            logits: [B, image_token_len, vocab_size]
        """
        B = audio_tokens.size(0)
        # Ensure audio_features have fixed length (interpolate if necessary)
        if audio_tokens.size(1) != self.audio_token_len:
            # print("Audio features size is less than audio token length !!!")
            audio_tokens = F.interpolate(audio_tokens.transpose(1,2), size=self.audio_token_len, mode='linear', align_corners=False)
            audio_tokens = audio_tokens.transpose(1,2)
        audio_emb = self.audio_proj(audio_tokens) + self.audio_pos_emb  # [B, audio_token_len, latent_dim]
        
        # For inference, initialize with learned image positional embeddings
        gen_img_emb = self.image_pos_emb.repeat(B, 1, 1)
    
        # Transformer expects [sequence, batch, features]
        gen_img_emb = gen_img_emb.transpose(0, 1)  # [image_token_len, B, latent_dim]
        audio_emb = audio_tokens.transpose(0, 1)  # [audio_token_len, B, latent_dim]
        
        # Create causal mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(gen_img_emb.size(0)).to(gen_img_emb.device)
        transformer_output = self.transformer_decoder(gen_img_emb, audio_emb, tgt_mask=tgt_mask)
        logits = self.token_logits(transformer_output.transpose(0,1))  # [B, image_token_len, vocab_size]
        side = int(self.image_token_len ** 0.5)
        if side * side != self.image_token_len:
            raise ValueError("image_token_len must be a perfect square")
        # For decoding, we use the predicted indices (via argmax).
        return logits, side

# Combined model that connects the audio encoder and the transformer.
class AudioToImageTokensModel(nn.Module):
    def __init__(self, audio_encoder, audio_to_token_transformer):
        super(AudioToImageTokensModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.audio_to_token_transformer = audio_to_token_transformer
    
    def forward(self, spectrogram):
        # Encode the spectrogram into audio tokens.
        audio_features = self.audio_encoder(spectrogram)  # [B, tokens_audio, latent_dim]
        logits = self.audio_to_token_transformer(audio_features)
        return logits