import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def plot_training_history(checkpoint_dir, save_path=None):
    """Plot training and validation loss and accuracy from checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    
    if len(checkpoint_files) == 0:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    epochs = []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for ckpt_file in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            epochs.append(checkpoint['epoch'] + 1)
            
            # Try new format first (train_metrics/val_metrics)
            train_metrics = checkpoint.get('train_metrics', checkpoint.get('metrics', {}))
            val_metrics = checkpoint.get('val_metrics', {})
            
            train_losses.append(train_metrics.get('loss', 0))
            train_accs.append(train_metrics.get('bit_accuracy', 0) * 100)
            
            if val_metrics:
                val_losses.append(val_metrics.get('loss', 0))
                val_accs.append(val_metrics.get('bit_accuracy', 0) * 100)
        except:
            continue
    
    if not epochs:
        print("No valid checkpoint data")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', label='Train Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', linewidth=2, marker='s', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, marker='o', label='Train Accuracy')
    if val_accs:
        ax2.plot(epochs, val_accs, 'r-', linewidth=2, marker='s', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    print(f"Final Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.2f}%")
    if val_losses:
        print(f"Final Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.2f}%")

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
