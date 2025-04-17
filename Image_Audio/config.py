# Define the config dictionary
config = {
    "dataset_params": {
        "train_dir": "/home/project/dataset/VGG_Sound_extracted/train_data",
        "test_dir": "/home/project/dataset/VGG_Sound_extracted/train_data",
        "drop_background_prob": 0.1,
        "drop_color_prob": 0.1
    },
    "audio_vae_params": {
        "vae_num_embeddings": 1024,
        "vae_embedding_dim": 512,
    },
    "image_vae_params": {
        "vae_num_embeddings": 1024,
        "vae_embedding_dim": 512,
    },    
    "model_params": {
        "dalle_spect_size": 2816,
        "dalle_image_size": 128,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 1024,
        "embd_dim": 512,
        "n_audio_tokens": 20*22,
        "n_image_tokens": 16*16
    },
    "train_params": {
        "task_name": "default",
        "batch_size": 32,
        "dalle_batch_size": 32,
        "num_epochs": 50,
        "num_epochs_dalle": 250,
        "dalle_image_loss": 10,
        "kl_weight": 0,
        "lr": 0.0001,
        "crit": "l1",
        "seed": 1111,
        "save_vae_training_image": True,
        "vae_ckpt_name": "vae_ckpt_128.pth",
        "dalle_ckpt_name": "dalle_ckpt_64.pth"
    }
}
