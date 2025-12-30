from TemporalDataset import TemporalDataset
from SeqSet import SeqSet
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models import UNet
from torch.nn import MSELoss
from torch.optim import Adam
import torch
import os
from datetime import datetime
import json
from CheckpointManager import CheckpointManager


BATCH_SIZE = 64
NUM_EPOCHS = 25
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


EXP_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXPERIMENT_DIR = f"./experiments/{EXP_ID}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_DIR, 'checkpoints'), exist_ok=True)
VIS_DIR = 'visualizations'
os.makedirs(os.path.join(EXPERIMENT_DIR, VIS_DIR), exist_ok=True)

def visualize_predictions(noisy_curr, pred_noise, true_noise, epoch, batch_idx, split='train'):
    """
    Visualize the noisy image, predicted noise, true noise, and resulting denoised images.
    """
    # Move tensors to CPU and convert to numpy
    noisy_curr_np = noisy_curr[0].cpu().numpy().transpose(1, 2, 0).squeeze()
    pred_noise_np = pred_noise[0].cpu().detach().numpy().transpose(1, 2, 0).squeeze()
    true_noise_np = true_noise[0].cpu().numpy().transpose(1, 2, 0).squeeze()

    # Calculate denoised images by subtracting noise from noisy image
    # This is a simplified denoising - actual diffusion models use the noise schedule
    pred_denoised = noisy_curr_np - pred_noise_np
    true_denoised = noisy_curr_np - true_noise_np

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1: Images
    axes[0, 0].imshow(noisy_curr_np, cmap='gray')
    axes[0, 0].set_title('Noisy Current Frame')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_denoised, cmap='gray')
    axes[0, 1].set_title('Predicted Denoised')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(true_denoised, cmap='gray')
    axes[0, 2].set_title('True Denoised (Ground Truth)')
    axes[0, 2].axis('off')

    # Row 2: Noise
    axes[1, 0].imshow(pred_noise_np, cmap='gray')
    axes[1, 0].set_title('Predicted Noise')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(true_noise_np, cmap='gray')
    axes[1, 1].set_title('True Noise')
    axes[1, 1].axis('off')

    # Difference map
    noise_diff = np.abs(pred_noise_np - true_noise_np)
    axes[1, 2].imshow(noise_diff, cmap='hot')
    axes[1, 2].set_title('Noise Prediction Error')
    axes[1, 2].axis('off')

    plt.suptitle(f'Epoch {epoch+1} - {split.capitalize()} Batch {batch_idx}')
    plt.tight_layout()

    filename = f'{VIS_DIR}/epoch_{epoch+1:03d}_{split}_batch_{batch_idx:04d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved visualization: {filename}')


temporal_dataset = TemporalDataset('synth1')

temporal_dataset.prepare_dataset(masking_dec=0.15, train_split=0.8)

train_triplets = temporal_dataset.create_triplets(train=True, gap_weights='exponential')
val_triplets = temporal_dataset.create_triplets(train=False, gap_weights='exponential')

# dataset.visualise_triplet(train_triplets)

training_data = SeqSet(train_triplets, D=1000)
val_data = SeqSet(val_triplets, D=1000)

train_loader = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle=True)

colourway = 'grayscale'

if colourway == 'grayscale':
    image_channels = 1

model = UNet(image_channels=image_channels, cond_embed_dim=256).to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-4)

loss = MSELoss()

history = {
    'train_loss' : [],
    'val_loss' : []
}

best_val_loss = float('inf')

Checkpoints = CheckpointManager(exp_id=EXP_ID, 
                                exp_dir=EXPERIMENT_DIR,
                                model=model, 
                                optimizer=optimizer)


for epoch in range(NUM_EPOCHS):

    epoch_total_train_loss = 0
    epoch_total_val_loss = 0
    
    model.train()

    for train_batch_idx, train_batch in enumerate(train_loader):

        noisy_curr = train_batch['noisy_curr'].to(DEVICE)
        prev_frame = train_batch['prev_frame'].to(DEVICE)
        d_level = train_batch['d_level'].to(DEVICE)
        delta_t = train_batch['delta_t'].to(DEVICE)
        noise = train_batch['noise'].to(DEVICE)



        pred_noise = model(noisy_curr=noisy_curr,
                        prev_frame=prev_frame,
                        d_level=d_level,
                        delta_t = delta_t)


        mse_train = loss(pred_noise, noise) # input, target
        epoch_total_train_loss += mse_train.item()

        optimizer.zero_grad()
        mse_train.backward()
        optimizer.step()

        # Visualize first batch of each epoch
        if train_batch_idx == 0:
            visualize_predictions(noisy_curr, pred_noise, noise, epoch, train_batch_idx, split='train')

        model.eval()
        
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):

            noisy_curr = val_batch['noisy_curr'].to(DEVICE)
            prev_frame = val_batch['prev_frame'].to(DEVICE)
            d_level = val_batch['d_level'].to(DEVICE)
            delta_t = val_batch['delta_t'].to(DEVICE)
            noise = val_batch['noise'].to(DEVICE)


            pred_noise = model(noisy_curr=noisy_curr,
                            prev_frame=prev_frame,
                            d_level=d_level,
                            delta_t = delta_t)

            mse_val = loss(pred_noise, noise) # input, target
            epoch_total_val_loss += mse_val.item()

            # Visualize first batch of validation
            if val_batch_idx == 0 and train_batch_idx % 5 == 0:
                visualize_predictions(noisy_curr, pred_noise, noise, epoch, val_batch_idx, split='val')

    epoch_train_loss = epoch_total_train_loss / len(train_loader)
    epoch_val_loss = epoch_total_val_loss / len(val_loader)

    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)

    is_best = epoch_val_loss < best_val_loss
    
    if is_best: 
        best_val_loss = epoch_val_loss

    Checkpoints.save_checkpoint(epoch=epoch, 
                                train_loss=mse_train, 
                                val_loss=mse_val, 
                                is_best=is_best)
    
    with open(f"{EXPERIMENT_DIR}/history.json", 'w') as f:
        json.dump(history, f)
    
    # Plot losses
    Checkpoints.plot_losses(history, f"{EXPERIMENT_DIR}/loss_curve.png")


    print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    


        
