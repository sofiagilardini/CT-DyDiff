
import json 
from datetime import datetime
import torch
import matplotlib.pyplot as plt 


class CheckpointManager():

    def __init__(self, exp_id:str, exp_dir:str, model, optimizer):
        
        self.exp_id = exp_id
        self.model = model
        self.optimizer = optimizer
        self.exp_dir = exp_dir

    
    def save_checkpoint(self, 
                        epoch, 
                        train_loss, 
                        val_loss, 
                        is_best=False):
        
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(), 
            'optimizer_state_dict' : self.optimizer.state_dict(), 
            'train_loss' : train_loss, 
            'val_loss' : val_loss
        }

        torch.save(checkpoint, f"{self.exp_dir}/checkpoints/best.pt")
        print(f"New best model saved. Val loss {val_loss:.4f}")

    
    def load_checkpoint(self, path, loading):

        if loading:

            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['val_loss']
    
    def plot_losses(self, history, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

        



