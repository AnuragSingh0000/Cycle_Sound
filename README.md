
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

## Training & Generation Workflows

### Audio → Image Pipeline

#### Training
1. **Pre‑train Image VAE**  
   - Train a discrete VAE on real images to learn a codebook of visual tokens.  
   - Encoder compresses images into a grid of code indices; decoder reconstructs the image from those indices.  
   - Losses: reconstruction (pixel or perceptual) + Gumbel‑Softmax KL regularization.

2. **Train Audio→Image Transformer**  
   - Freeze the pretrained Image VAE.  
   - For each training pair (audio clip → corresponding image):  
     - Convert audio clip to a Mel‑spectrogram.  
     - Encode spectrogram through the Audio Encoder → sequence of latent audio tokens.  
     - Use the Image VAE encoder to obtain “ground‑truth” image token indices.  
     - Transformer Decoder learns to map audio tokens + positional embeddings to the image token sequence.  
   - Loss: cross‑entropy between predicted token logits and ground‑truth code indices, optionally augmented with a VAE reconstruction loss (decoded image vs. real).

#### Generation
1. **Encode Audio**  
   - Input a new audio clip → compute Mel‑spectrogram → Audio Encoder → audio token sequence.
2. **Predict Image Tokens**  
   - Transformer Decoder autoregressively generates the full image token sequence conditioned on audio tokens.
3. **Decode to Image**  
   - Convert predicted token indices back through the Image VAE decoder to produce the final RGB image.

---

### Image → Audio Pipeline

#### Training
1. **Pre‑train Audio VAE**  
   - Train a discrete VAE on Mel‑spectrograms extracted from audio clips.  
   - Encoder compresses each spectrogram into discrete audio tokens; decoder reconstructs spectrogram + waveform.

2. **Train Image→Audio Transformer**  
   - Freeze the pretrained Audio VAE.  
   - For each training pair (image → corresponding audio clip):  
     - Resize and encode image through the Image Encoder → sequence of image tokens.  
     - Use the Audio VAE encoder to obtain “ground‑truth” audio token indices.  
     - Transformer Decoder learns to map image tokens + positional embeddings to the audio token sequence.  
   - Loss: cross‑entropy between predicted logits and true audio code indices, optionally with reconstruction loss on the decoded spectrogram.

#### Generation
1. **Encode Image**  
   - Input a new image → Image Encoder → discrete image token sequence.
2. **Predict Audio Tokens**  
   - Transformer Decoder autoregressively generates the full audio token sequence conditioned on image tokens.
3. **Decode to Spectrogram & Waveform**  
   - Convert tokens through the Audio VAE decoder to reconstruct a Mel‑spectrogram.  
   - Apply an inverse Mel transform (via librosa) to convert the spectrogram back to a time‑domain waveform.

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



