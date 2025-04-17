import os
import glob
import cv2
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Enforce GPU-only execution
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This code requires a GPU.")
device = torch.device("cuda")


# -----------------------------
# Dataset for loading image and audio spectrograms.
# -----------------------------
class ImageAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, sr=22050, n_mels=64, duration=1.0, hop_length=256):
        """
        Args:
            root_dir (str): Directory containing .jpg and .mp3 pairs.
            transform (callable, optional): Transform to be applied on the image sample.
            sr (int): Audio sample rate.
            n_mels (int): Number of Mel bands.
            duration (float): Duration of the audio clip in seconds.
            hop_length (int): Hop length for spectrogram computation.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.hop_length = hop_length
        self.expected_length = int(sr * duration)

        # Gather all pairs of .jpg and .mp3 that share a base filename, sorted for consistency.
        self.files = []
        for img_path in sorted(glob.glob(os.path.join(root_dir, "*.jpg"))):
            base = os.path.splitext(img_path)[0]
            mp3_path = base + ".mp3"
            if os.path.exists(mp3_path):
                self.files.append((img_path, mp3_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, mp3_path = self.files[idx]

        # ---- Load image ----
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            # Default transform: convert to tensor and normalize to [0,1]
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # ---- Load audio and convert to spectrogram ----
        y, _ = librosa.load(mp3_path, sr=self.sr)  # librosa can load .mp3
        if len(y) < self.expected_length:
            y = np.pad(y, (0, self.expected_length - len(y)), mode='constant')
        else:
            y = y[:self.expected_length]

        # Compute Mel spectrogram.
        spect = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        # Convert power spectrogram to decibel (dB) units.
        spect_db = librosa.power_to_db(spect, ref=np.max)
        # Normalize: assume dB values in [-80, 0]. Scale to [0, 1]
        spect_norm = (spect_db + 80) / 80.0
        spect_tensor = torch.tensor(spect_norm, dtype=torch.float32).unsqueeze(0)
        # Pad time dimension so that T is divisible by 8 (for autoencoder consistency).
        T_dim = spect_tensor.shape[-1]
        if T_dim % 8 != 0:
            pad_len = 8 - (T_dim % 8)
            spect_tensor = F.pad(spect_tensor, (0, pad_len), mode='constant', value=0)


        return {
            "image": img,                # Tensor: (3, H, W)
            "audio_spect": spect_tensor, # Tensor: (1, n_mels, time_frames)
            "audio_waveform": y,         # Raw waveform (numpy array)
            "sr": self.sr               # Sample rate
        }
