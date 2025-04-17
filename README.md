```markdown
# Cycle_Sound

## Overview

**Cycle_Sound** implements two complementary cross‑modal generation pipelines in PyTorch:

1. **Audio → Image**: Convert short audio clips (as Mel‑spectrograms) into realistic images.
2. **Image → Audio**: Convert images into reconstructed audio spectrograms and waveforms.

Each direction uses a discrete VAE (via Gumbel‑Softmax quantization) to map inputs into a codebook, and a Transformer decoder to predict target tokens in the other modality.

## Novelty & Contributions

- **Bidirectional Cycle Framework**: By training audio→image and image→audio end‑to‑end (separately), the repository enables cyclic consistency experiments for cross‑modal generation.
- **Discrete Codebook Quantization**: Both audio and image modalities share the core idea of learning a discrete latent space, which stabilizes generation and enables efficient token‑level modeling.
- **Transformer Decoder for Cross‑Modal Mapping**: Rather than a heavy encoder–decoder network, the approach uses a lightweight transformer decoder conditioned on source tokens and positional embeddings to predict target tokens.
- **Modular Design**: Clear separation between VAE modules (for quantization) and Transformer modules (for sequence modeling) allows independent improvement or replacement.

## Model Architectures

### 1. DiscreteVAE (Image VAE)

- **Encoder**: Stacked convolutional blocks + residual layers → projects image to a feature map of size `(C=num_embeddings, H', W')`, where `num_embeddings` = codebook size.
- **Quantizer**: Applies Gumbel‑Softmax over the channel axis to produce a soft one‑hot map → samples discrete indices → uses them to index learned embedding vectors.
- **Decoder**: Transposed convolutions + residual upsampling → reconstructs RGB images.

**Forward pass**:
```python
logits = encoder(x)         # [B, num_embeddings, H', W']
sampled, kl, log_qy = quantizer(logits)
out = decoder(sampled)       # [B, 3, H, W]
```  
**Codebook indices** are obtained via `argmax` on `logits`.

### 2. AudioEncoder & Audio→Image Transformer

- **Audio Encoder**: 4× downsampling conv2d layers + BatchNorm + ReLU → outputs `[B, latent_dim, H_s, W_s]`, flattened to `[B, tokens_audio, latent_dim]`.
- **Positional Embedding**: Learned positional vectors for audio tokens and target image tokens.
- **Transformer Decoder**: Causal self‑attention on target embeddings, cross‑attention to audio embeddings.
- **Output Projection**: Linear layer mapping decoder outputs to logits over the VAE codebook (`vocab_size`).

**Generation**:
```python
audio_tokens = audio_encoder(spectrogram)             # [B, T_audio, D]
img_pos = image_pos_emb.repeat(B,1,1)                  # [B, T_image, D]
logits = transformer_decoder(img_pos, audio_tokens)   # [B, T_image, vocab]
pred_indices = logits.argmax(-1)                       # [B, T_image]
```  
Decode indices with Image VAE to produce final images.

### 3. DiscreteVAE (Audio VAE)

- **Encoder**: Conv2d blocks → projects spectrogram to `[B, num_embeddings, H', W']`.
- **Quantizer**: Gumbel‑Softmax one‑hot over channels → discrete audio tokens.
- **Decoder**: Transposed conv2d → reconstruct Mel‑spectrogram.
- **Waveform Reconstruction**: Inverse Mel transform via `librosa`.

### 4. ImageEncoder & Image→Audio Transformer

- **Image Encoder**: Similar to Image VAE’s encoder but stops before quantization.
- **Quantizer + Flatten**: Argmax to get discrete indices → flatten to sequence `[B, T_image, D]`.
- **Transformer Decoder**: Causal decoder predicts audio token logits given image tokens.
- **Audio Decoder**: Uses discrete tokens to reconstruct spectrogram and waveform.

## Repository Structure

```bash
anuragsingh0000-cycle_sound/
├── VGG_Sound_data_extraction/  # Download & preprocess VGG‑Sound
├── Audio_Image/                # Audio→Image pipeline
├── Image_Audio/                # Image→Audio pipeline
├── README.md                   # Setup & usage (this file)
└── requirements.txt            # Dependencies
```

## Prerequisites

- Python 3.8+  •  CUDA‑enabled GPU  •  Linux/macOS

## Installation

```bash
git clone https://github.com/<username>/anuragsingh0000-cycle_sound.git
cd anuragsingh0000-cycle_sound
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

```bash
cd VGG_Sound_data_extraction
python download_and_extract.py --output_dir ../dataset/VGG_Sound_extracted
```

Ensure `dataset/VGG_Sound_extracted/{train_data,test_data}` contains matching `.jpg` + `.mp3` pairs.

## Configuration

Edit `Audio_Image/config.py` and `Image_Audio/config.py` to adjust:
- Dataset paths
- VAE & Transformer hyperparameters
- Training parameters (batch size, epochs, learning rate)

## Training

1. **Image VAE** (Audio→Image):
   ```bash
   cd Audio_Image
   python train_vae.py
   ```
2. **Audio→Image Transformer**:
   ```bash
   python train_model.py
   ```
3. **Image→Audio Transformer**:
   ```bash
   cd Image_Audio
   python train_model.py
   ```

## Inference

- **Images from Audio**:
  ```bash
  cd Audio_Image
  python generate.py
  ```
- **Audio from Images**:
  ```bash
  cd Image_Audio
  python generate.py
  ```

## Visualization & Outputs

- Generated images: `Audio_Image/final_gen_image/`
- Reconstructed audio: `Audio_Image/final_audio/`, `Image_Audio/generated/`
- Spectrogram plots: `saved_spectrograms/`

## Troubleshooting

- **CUDA**: Scripts abort if no GPU detected.
- **Memory**: Lower `batch_size` if OOM errors.
- **Filename Matching**: Ensure `.jpg` and `.mp3` share the same basename.

## License

MIT © 2025

## Acknowledgements

- **VGG‑Sound** dataset by Google Research
- Gumbel‑Softmax trick for discrete VAEs
- Transformer models for sequence modeling
```

