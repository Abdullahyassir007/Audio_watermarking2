import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def plot_training_history(checkpoint_dir, save_path=None):
    """Plot training loss from checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    
    if len(checkpoint_files) == 0:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    epochs, losses, msg_losses = [], [], []
    
    for ckpt_file in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            epochs.append(checkpoint['epoch'] + 1)
            metrics = checkpoint.get('metrics', {})
            losses.append(metrics.get('loss', 0))
            msg_losses.append(metrics.get('msg_loss', 0))
        except:
            continue
    
    if not epochs:
        print("No valid checkpoint data")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Total Loss')
    plt.plot(epochs, msg_losses, 'r-', linewidth=2, label='Message Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    print(f"Final Loss: {losses[-1]:.4f} | Best Loss: {min(losses):.4f}")

def plot_accuracy(bit_accuracy, save_path=None):
    """Plot bit accuracy"""
    plt.figure(figsize=(6, 5))
    plt.bar(['Bit Accuracy'], [bit_accuracy * 100], color='purple', alpha=0.7)
    plt.ylabel('Accuracy (%)')
    plt.title('Watermark Bit Accuracy')
    plt.ylim([0, 105])
    plt.axhline(y=100, color='green', linestyle='--', label='Perfect')
    plt.axhline(y=50, color='red', linestyle='--', label='Random')
    plt.text(0, bit_accuracy * 100, f'{bit_accuracy*100:.2f}%', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()
    
    plot_training_history(args.checkpoint_dir, args.save)
