import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import ImageAudioDataset
from Audio_model import AudioEncoder, AudioToImageTokensModel, AudioToTokenTransformer
from Image_vae import DiscreteVAE
from config import config
from visualize import plot_img, visualize_spectrogram, show_test_audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_for_one_epoch(epoch_idx, model, vae, loader, optimizer, config):
    model.train()
    losses = []
    
    # Assume your discrete VAE has an 'encode' method that returns token indices
    for data in tqdm(loader, desc=f"Epoch {epoch_idx+1}"):
        # Get batch data and move to device
        images = data['image'].float().to(device)         # shape: [B, 3, H, W]
        spectrograms = data['audio_spect'].float().to(device)  # shape: [B, 1, H_spec, W_spec]
        optimizer.zero_grad()
        with torch.no_grad():
            target_tokens = vae.get_codebook_indices(images)
            target_tokens = target_tokens.view(target_tokens.size(0), -1)
        
        logits, side = model(spectrograms)
        B, image_token_len, vocab_size = logits.shape
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_tokens.view(-1))
        
        # For decoding, use argmax to get predicted token indices.
        # predicted_indices = torch.argmax(logits, dim=-1)  # shape: [B, image_token_len]
        probs = F.softmax(logits, dim=-1)
        predicted_indices = torch.multinomial(probs.view(-1, vocab_size), 1).view(B, image_token_len)
        
        # Reshape to grid: [B, side, side]
        predicted_indices_grid = predicted_indices.view(B, side, side)
        
        # Convert indices to one-hot: output shape [B, vocab_size, side, side]
        predicted_one_hot = F.one_hot(predicted_indices_grid, num_classes=vocab_size).permute(0, 3, 1, 2).float()
        
        # Now decode using the VAE.
        generated_images = vae.decode_from_codebook_indices(predicted_one_hot)

        mse = nn.MSELoss()
        recon_loss = mse(images, generated_images)

        # p_loss = perceptual_loss(generated_images, images)
        loss += recon_loss

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    mean_loss = np.mean(losses)
    print(f'Finished epoch: {epoch_idx + 1} | Loss: {mean_loss:.4f}')
    return mean_loss

def train(config: dict, model, vae):
    # Set random seeds for reproducibility
    torch.manual_seed(config["train_params"]["seed"])
    np.random.seed(config["train_params"]["seed"])
    random.seed(config["train_params"]["seed"])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config["train_params"]["seed"])
    
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
    vae_checkpoint_path = f"{config['train_params']['task_name']}/{config['train_params']['vae_ckpt_name']}"
    if os.path.exists(vae_checkpoint_path):
        print('Found checkpoint... Loading VAE')
        vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    else:
        print(f'No checkpoint found at {vae_checkpoint_path}... Exiting')
        return
    vae.to(device)
    vae.eval()  # Freeze VAE weights
    for param in vae.parameters():
        param.requires_grad = False


    
    model_checkpoint_path = os.path.join(config["train_params"]["task_name"], "audio_to_image_model.pth")
    if os.path.exists(model_checkpoint_path):
        print('Found checkpoint... Loading Audio to Image Model')
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))

    # Define optimizer on the parameters of the audio-to-image model.
    optimizer = Adam(model.parameters(), lr=config["train_params"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    
    best_loss = np.inf
    
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, vae, dataloader, optimizer, config)
        scheduler.step(mean_loss)
        
        # Save model if improvement is observed.
        if mean_loss < best_loss:
            print(f'Improved Loss from {best_loss:.4f} to {mean_loss:.4f}... Saving Model')
            torch.save(model.state_dict(), model_checkpoint_path)
            best_loss = mean_loss
        else:
            print(f'No Loss Improvement. Best Loss: {best_loss:.4f}')

        # Generate and display images every few epochs.
        if (epoch_idx) % 1 == 0:
            # Use a batch from the dataloader for inference.
            sample_data = next(iter(dataloader))
            spectrograms = sample_data['audio_spect'].float().to(device)
            sample_img = sample_data['image']
            
            with torch.no_grad():
                target_tokens = vae.get_codebook_indices(sample_img)
                target_tokens = target_tokens.view(target_tokens.size(0), -1)
                logits, side = model(spectrograms)
                B, image_token_len, vocab_size = logits.shape
                predicted_indices = torch.argmax(logits, dim=-1)  # shape: [B, image_token_len]
                # Reshape to grid: [B, side, side]
                predicted_indices_grid = predicted_indices.view(B, side, side)
                # Convert indices to one-hot: output shape [B, vocab_size, side, side]
                predicted_one_hot = F.one_hot(predicted_indices_grid, num_classes=vocab_size).permute(0, 3, 1, 2).float()
                # Now decode using the VAE.
                generated_images = vae.decode_from_codebook_indices(predicted_one_hot)
                    
            show_test_audio(sample_data, epoch=epoch_idx%3)
            plot_img(sample_img, epoch_idx%3, "Real_image", save_dir="saved_images_model")
            plot_img(generated_images, epoch_idx%3, "Generated_image", save_dir="saved_images_model")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

    # Initialize your models:
    audio_encoder = AudioEncoder(in_channels=1, base_channels=64, latent_dim=config["audio_params"]["embd_dim"])
    
    audio_to_token_transformer = AudioToTokenTransformer(
        latent_dim=config["audio_params"]["embd_dim"], nhead=config["audio_params"]["n_head"], num_layers=config["audio_params"]["n_layer"], 
        audio_token_len=64, image_token_len=(config["model_params"]["dalle_image_size"]//8)**2, vocab_size=config["audio_params"]["n_embd"]
    )
    model = AudioToImageTokensModel(audio_encoder, audio_to_token_transformer).to(device)
    
    # Initialize the discrete VAE (for images)
    vae = DiscreteVAE(
        num_embeddings=config["model_params"]["vae_num_embeddings"],
        embedding_dim=config["model_params"]["vae_embedding_dim"]
    )
    
    # Start training
    train(config, model, vae)
