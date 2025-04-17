import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import librosa
import scipy.signal
import scipy.io.wavfile as wavfile
from IPython.display import Audio, display
from config import config
from dataset import ImageAudioDataset
from Image_encoder import ImageEncoder
from Audio_VAE import AudioAE
from Image_model import ImagetoAudio, ImagetoAudioTransformer
from visualize import show_test_audio, visualize_spectrogram, plot_img

# ─── Ensure device is a torch.device, not a string ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = ("cuda" if torch.cuda.is_available() else "cpu")

def train(model, aae, loader, optimizer, scheduler, epoch=50):
    losses = []
    for e in range(epoch):
        batch_idx = -1
        for batch in tqdm(loader):
            batch_idx += 1
            image = batch["image"].to(device)
            spect = batch["audio_spect"].to(device)
            optimizer.zero_grad()
            
            audio_tokens = aae.get_codebook_indices(spect)
            audio_tokens.reshape(audio_tokens.shape[0], -1)
            logits = model(image)
            B, audio_token_len, vocab_size = logits.shape
            loss = F.cross_entropy(logits.permute(0, 2, 1), audio_tokens.reshape(B, -1).long())
            
            if (batch_idx==0 and e%1==0):
                # For decoding, use argmax to get predicted token indices.
                recon = aae.decode_from_logits(logits)
                
                save_dir = "reconsturcted"
                save_dir2 = "given"
                
                # Visualize spectrograms
                visualize_spectrogram(spect, epoch=e%3, save_dir=save_dir2)
                visualize_spectrogram(recon, epoch=e%3, save_dir=save_dir)
                sr = batch["sr"]
                plot_img(image, epoch=e%3, save_dir=save_dir2)
                
                spect = spect.detach().cpu().squeeze(1).numpy()
                recon = recon.detach().cpu().squeeze(1).numpy()
                for i in range(4):
                    
                    spect_db = spect[i] * 80.0 - 80.0
                    # Convert dB to power
                    power = librosa.db_to_power(spect_db)
                
                    # Reconstruct audio using inverse mel-spectrogram
                    audio = librosa.feature.inverse.mel_to_audio(
                        power,
                        sr=sr[i].item(),
                        n_fft=1024,
                        hop_length=256,
                        window=scipy.signal.windows.hann
                    )
                
                    filename = os.path.join(save_dir2, f"epoch_{e%3}_sample_{i+1}_real.wav")
                    wavfile.write(filename, sr[i].item(), audio)
                    
                    pred_db = recon[i] * 80.0 - 80.0
                    # Convert dB to power
                    pred_power = librosa.db_to_power(pred_db)
                
                    # Reconstruct audio using inverse mel-spectrogram
                    audio = librosa.feature.inverse.mel_to_audio(
                        pred_power,
                        sr=sr[i].item(),
                        n_fft=1024,
                        hop_length=256,
                        window=scipy.signal.windows.hann
                    )
                    filename = os.path.join(save_dir, f"epoch_{e%3}_sample_{i+1}_gen.wav")
                    wavfile.write(filename, sr[i].item(), audio)
                    
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        mean_loss = np.mean(losses)
        scheduler.step(mean_loss) 
        print(f'Finished epoch: {e + 1} | Loss: {mean_loss:.4f}')
        torch.save(model.state_dict(), os.path.join(config["train_params"]["task_name"], config["train_params"]["dalle_ckpt_name"]))
    
    
os.makedirs(config["train_params"]["task_name"], exist_ok=True)
    
num_epochs = config["train_params"]["num_epochs_dalle"]
    
# Image transformation: ensure images are resized to the size expected by the VAE.
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((config["model_params"]["dalle_image_size"], config["model_params"]["dalle_image_size"])),
    T.ToTensor()
])

# Dataset and DataLoader
root_dir = config["dataset_params"]["train_dir"]
dataset = ImageAudioDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load the pre-trained discrete VAE for image encoding
aae = AudioAE(config["audio_vae_params"]["vae_num_embeddings"], config["audio_vae_params"]["vae_embedding_dim"]).to(device)
aae_checkpoint_path = f"{config['train_params']['task_name']}/{config['train_params']['vae_ckpt_name']}"
if os.path.exists(aae_checkpoint_path):
    print('Found checkpoint... Loading VAE')
    aae.load_state_dict(torch.load(aae_checkpoint_path, map_location=device))
else:
    print(f'No checkpoint found at {aae_checkpoint_path}... Exiting')
    
aae.to(device)
aae.eval()  # Freeze VAE weights
for param in aae.parameters():
    param.requires_grad = False

image_encoder = ImageEncoder(config["image_vae_params"]["vae_num_embeddings"], config["image_vae_params"]["vae_embedding_dim"]).to(device)
image_to_audio_transformer = ImagetoAudioTransformer(config["model_params"]["embd_dim"], nhead=config["model_params"]["n_head"], num_layers=config["model_params"]["n_layer"], 
                audio_token_len=config["model_params"]["n_audio_tokens"], image_token_len=config["model_params"]["n_image_tokens"], vocab_size=config["model_params"]["n_embd"])
image_encoder.train()

model = ImagetoAudio(image_encoder, image_to_audio_transformer)
model_checkpoint_path = os.path.join(config["train_params"]["task_name"], config["train_params"]["dalle_ckpt_name"])
if os.path.exists(model_checkpoint_path):
    print('Found checkpoint... Loading Audio to Image Model')
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.to(device)
model.train()

# optimizer = Adam(list(image_encoder.parameters()) + list(model.parameters()), lr=config["train_params"]["lr"])
optimizer = Adam((model.parameters()), lr=config["train_params"]["lr"])
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

train(model, aae, dataloader, optimizer, scheduler, epoch=12)
torch.save(image_encoder.state_dict(), os.path.join(config["train_params"]["task_name"], "image_vae.pth"))
