import torch
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from visualize import plot_img, show_test_audio, visualize_spectrogram
from config import config
from Image_vae import DiscreteVAE
from dataset import ImageAudioDataset

device = torch.device('cuda')


def perceptual_loss(input_img, output_img):
    # Expect input_img, output_img to be in range [0, 1], shape (B, C, H, W)
    input_features = feature_extractor(input_img)
    output_features = feature_extractor(output_img)
    return F.mse_loss(output_features['feat_relu4_1'], input_features['feat_relu4_1'])

def train_for_one_epoch_vae(epoch_idx, model, mnist_loader, optimizer, criterion, config):
    losses = []
    count = 0
    for data in tqdm(mnist_loader):
        im = data['image'].float().to(device)
        optimizer.zero_grad()
        output, kl, log_qy = model(im)
        
        if epoch_idx % 1 == 0 and count % 100 == 0:
            plot_img(im, epoch_idx%10, "Real Image")
            plot_img(output, epoch_idx%10, "Generated Image")
        
        # Basic reconstruction loss
        recon_loss = criterion(output, im)
        
        # Perceptual loss
        perc_loss = perceptual_loss(im, output)

        # Total loss: blend both
        # loss = recon_loss
        loss = (40 * recon_loss + perc_loss)
        
        losses.append(loss.item())
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        count += 1
        
    print(f"Finished epoch: {epoch_idx + 1} | Loss : {np.mean(losses):.4f}")
    return np.mean(losses)

def train_vae():
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    
    num_epochs = config['train_params']['num_epochs']
    model = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    ).to(device)
    
    checkpoint_path = f"{config['train_params']['task_name']}/{config['train_params']['vae_ckpt_name']}"
    if os.path.exists(checkpoint_path):
        print("Found checkpoint... Starting training from that")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config["model_params"]["dalle_image_size"], config["model_params"]["dalle_image_size"])),
        T.ToTensor()
    ])

    root_dir = config["dataset_params"]["train_dir"]
    ds = ImageAudioDataset(root_dir, transform=transform)
    dataloader = DataLoader(ds, batch_size=8, shuffle=True)
    
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    criterion = {"l1": torch.nn.SmoothL1Loss(beta=0.1), "l2": torch.nn.MSELoss()}.get(config['train_params']['crit'])
    
    best_loss = np.inf
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch_vae(epoch_idx, model, dataloader, optimizer, criterion, config)
        scheduler.step(mean_loss)
        
        if mean_loss < best_loss:
            print(f"Improved Loss from {best_loss:.4f} to {mean_loss:.4f} .... Saving Model")
            torch.save(model.state_dict(), checkpoint_path)
            best_loss = mean_loss
        else:
            print(f"No Loss Improvement. Best Loss : {best_loss:.4f}")



if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # Load VGG16 once and extract intermediate layers
    vgg = vgg16(pretrained=True).to(device).eval()
    return_nodes = {'features.16': 'feat_relu4_1'}  # relu4_1 is commonly used
    feature_extractor = create_feature_extractor(vgg, return_nodes=return_nodes)
    
    # Freeze VGG parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False

    train_vae()
