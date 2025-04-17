import torch
from torch.utils.data import DataLoader
import librosa
import os
import scipy 
from tqdm.auto import tqdm
import scipy.io.wavfile as wavfile
import scipy.signal
from IPython.display import Audio, display
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from Audio_VAE import AudioAE
from visualize import visualize_spectrogram, plot_img, show_test_audio
from config import config
from dataset import ImageAudioDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)



def perceptual_loss(input_img, output_img):
    # Expect input_img, output_img to be in range [0, 1], shape (B, C, H, W)
    input_features = feature_extractor(input_img)
    output_features = feature_extractor(output_img)
    return F.mse_loss(output_features['feat_relu4_1'], input_features['feat_relu4_1'])

# ---- Training ----
def train_audio_autoencoder(model, dataloader, num_epochs=11, lr=1e-4, save_path="audio_ae.pth"):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_idx = 0
        for batch in tqdm(dataloader):
            batch_idx += 1
            # Get spectrogram [B, 1, n_mels, time_frames]
            spectrogram = batch['audio_spect'].to(device).float()
            optimizer.zero_grad()
            recon = model(spectrogram)
            loss = criterion(recon, spectrogram)
            # loss += perceptual_loss(recon, spectrogram)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (batch_idx % 100 == 0 and epoch % 1 == 0):
                save_dir = "reconsturcted"
                os.makedirs(save_dir, exist_ok=True)
                
                # Visualize spectrograms
                visualize_spectrogram(spectrogram, epoch%3)
                visualize_spectrogram(recon, epoch%3, save_dir=save_dir)
                
                # Convert predictions from [0, 1] back to dB scale, then power
                pred_spect = recon.detach().cpu().squeeze(1).numpy()  # [B, n_mels, T]
                sr = batch["sr"]  # list of sample rates per sample
                
                
                for i in range(min(4, pred_spect.shape[0])):
                    # Undo normalization: [0, 1] → [-80, 0] dB
                    pred_db = pred_spect[i] * 80.0 - 80.0
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
                    filename = os.path.join(save_dir, f"epoch_{epoch%3}_sample_{i+1}.wav")
                    wavfile.write(filename, sr[i].item(), audio)
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        if save_path:
            torch.save(model.state_dict(), save_path)



# Load VGG16 once and extract intermediate layers
vgg = vgg16(pretrained=True).to(device).eval()
return_nodes = {'features.16': 'feat_relu4_1'}  # relu4_1 is commonly used
feature_extractor = create_feature_extractor(vgg, return_nodes=return_nodes)

# Freeze VGG parameters
for param in feature_extractor.parameters():
    param.requires_grad = False

# Image transform (used even if not needed here)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((config["model_params"]["dalle_image_size"], config["model_params"]["dalle_image_size"])),
    T.ToTensor()
])

# Dataset and DataLoader
root_dir = config["dataset_params"]["train_dir"]
dataset = ImageAudioDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


if not os.path.exists(config['train_params']['task_name']):
    os.mkdir(config['train_params']['task_name'])    
num_epochs = config['train_params']['num_epochs']

# ---- Model Initialization ----
# Create an instance of the full AudioAE model. 
model = AudioAE(config["audio_vae_params"]["vae_num_embeddings"], config["audio_vae_params"]["vae_embedding_dim"]).to(device)
audio_ae_path = f"{config['train_params']['task_name']}/{config['train_params']['vae_ckpt_name']}"
if os.path.exists(audio_ae_path):
    print("Found checkpoint... Starting training from that")
    model.load_state_dict(torch.load(audio_ae_path, map_location=device))

# Start training the AudioAE model.
train_audio_autoencoder(
    model=model,
    dataloader=dataloader,
    num_epochs=10,
    lr=1e-4,
    save_path=f"{config['train_params']['task_name']}/{config['train_params']['vae_ckpt_name']}"
)

