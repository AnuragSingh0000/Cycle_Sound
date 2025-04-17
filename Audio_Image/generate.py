import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import ImageAudioDataset
from Audio_model import AudioEncoder, AudioToImageTokensModel, AudioToTokenTransformer
from Image_vae import DiscreteVAE
from config import config
from visualize import plot_img, visualize_spectrogram, show_test_audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate(test_loader, vae, model, l=0, r=10):
    # Load the pre-trained discrete VAE for image encoding
    vae_checkpoint_path = f"{config['train_params']['task_name']}/{config['train_params']['vae_ckpt_name']}"
    if os.path.exists(vae_checkpoint_path):
        print('Found checkpoint... Loading VAE')
        vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    else:
        print(f'No checkpoint found at {vae_checkpoint_path}... Exiting')
        return
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False


    
    model_checkpoint_path = os.path.join(config["train_params"]["task_name"], "audio_to_image_model.pth")
    if os.path.exists(model_checkpoint_path):
        print('Found checkpoint... Loading Audio to Image Model')
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    else:
        print(f'No checkpoint found at {model_checkpoint_path}... Exiting')
        return 
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for batch_idx, batch in enumerate(test_loader):
        if (batch_idx < l):
            continue
        elif (batch_idx > r):
            print("Generated the test results!!!")
            return
        img = batch["image"].float().to(device)
        spect = batch["audio_spect"].float().to(device)
        show_test_audio(batch, epoch=batch_idx, save_dir="final_audio")
        plot_img(img, epoch=batch_idx, save_dir="final_image")
        
        with torch.no_grad():
            logits, side = model(spect)
            B, image_token_len, vocab_size = logits.shape
            
            predicted_indices = torch.argmax(logits, dim=-1)
            predicted_indices_grid = predicted_indices.view(B, side, side)
            predicted_one_hot = F.one_hot(predicted_indices_grid, num_classes=vocab_size).permute(0, 3, 1, 2).float()
            
            generated_images = vae.decode_from_codebook_indices(predicted_one_hot)
            plot_img(generated_images, epoch=batch_idx, save_dir="final_gen_image")
            
            
if __name__ == '__main__':
    audio_encoder = AudioEncoder(in_channels=1, base_channels=64, latent_dim=config["audio_params"]["embd_dim"])
    audio_to_token_transformer = AudioToTokenTransformer(
        latent_dim=config["audio_params"]["embd_dim"], nhead=config["audio_params"]["n_head"], num_layers=config["audio_params"]["n_layer"], 
        audio_token_len=64, image_token_len=(config["model_params"]["dalle_image_size"]//8)**2, vocab_size=config["audio_params"]["n_embd"]
    )
    model = AudioToImageTokensModel(audio_encoder, audio_to_token_transformer).to(device)
    vae = DiscreteVAE(
        num_embeddings=config["model_params"]["vae_num_embeddings"],
        embedding_dim=config["model_params"]["vae_embedding_dim"]
    )

    print("Hello World")
    # Dataset and DataLoader
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config["model_params"]["dalle_image_size"], config["model_params"]["dalle_image_size"])),
        T.ToTensor()
    ])
    root_dir = config["dataset_params"]["test_dir"]
    dataset = ImageAudioDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    generate(dataloader, vae, model, 0, 24)