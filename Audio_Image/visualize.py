import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io.wavfile as wavfile

def show_test_audio(batch, sr=22050, epoch=0, save_dir="saved_test_audio"):
    audio_waveforms = batch["audio_waveform"]
    n = min(len(audio_waveforms), 4)
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    for i in range(n):
        audio_np = audio_waveforms[i].detach().cpu().numpy()
        # Normalize to int16 range [-32768, 32767] if needed
        audio_np = audio_np / np.max(np.abs(audio_np))  # Normalize to [-1, 1]
        audio_int16 = (audio_np * 32767).astype(np.int16)

        filename = os.path.join(save_dir, f"epoch_{epoch}_sample_{i+1}.wav")
        wavfile.write(filename, sr, audio_int16)


def plot_img(images, epoch=0, title="Image", save_dir="saved_images"):
    """Plot and save images from a batch."""
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(4, images.shape[0])  # Show up to 4 samples
    images = images.cpu().detach()  # Move to CPU and detach gradients

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))

    for i in range(num_images):
        img = images[i]  # Select image
        img = img.permute(1, 2, 0).numpy()  # Convert (C, H, W) → (H, W, C)

        # Normalize if needed (for images in range [-1, 1])
        img = (img - img.min()) / (img.max() - img.min())

        ax = axes[i] if num_images > 1 else axes
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{title} {i+1}")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()


def visualize_spectrogram(audio_spect, sr=22050, hop_length=256, save_dir="saved_spectrograms"):
    """Visualize and save spectrograms (assumed normalized to [0,1])."""
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(4, audio_spect.shape[0])
    
    for i in range(num_images):
        spect_np = audio_spect[i].detach().squeeze(0).cpu().numpy()
        plt.figure(figsize=(6, 4))
        img = librosa.display.specshow(spect_np, sr=sr, hop_length=hop_length,
                                    x_axis='time', y_axis='mel', cmap="magma")
        plt.title(f"Spectrogram {i+1}")
        plt.colorbar()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"spectrogram_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        
