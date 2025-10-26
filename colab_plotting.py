"""
Simple plotting code for Google Colab
Copy and paste these cells into your Colab notebook
"""

# ============================================
# CELL 1: Plot Training History
# ============================================
import matplotlib.pyplot as plt
import torch
from pathlib import Path

checkpoint_dir = "./checkpoints"  # Change this to your checkpoint directory
checkpoint_files = sorted(Path(checkpoint_dir).glob("*.pt"))

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

if epochs:
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', label='Total Loss')
    plt.plot(epochs, msg_losses, 'r-', linewidth=2, marker='s', label='Message Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Total Epochs: {len(epochs)}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Best Loss: {min(losses):.4f} at Epoch {epochs[losses.index(min(losses))]}")
else:
    print("No checkpoints found!")


# ============================================
# CELL 2: Plot Bit Accuracy (after evaluation)
# ============================================
import matplotlib.pyplot as plt

# Replace this with your actual bit accuracy from evaluation
bit_accuracy = 0.98  # Example: 98% accuracy

plt.figure(figsize=(6, 5))
plt.bar(['Bit Accuracy'], [bit_accuracy * 100], color='purple', alpha=0.7, edgecolor='black', width=0.5)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Watermark Bit Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0, 105])
plt.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect (100%)')
plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
plt.text(0, bit_accuracy * 100 + 2, f'{bit_accuracy*100:.2f}%', 
         ha='center', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"Bit Accuracy: {bit_accuracy*100:.2f}%")


# ============================================
# CELL 3: Quick Evaluation with Plot
# ============================================
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from models import INN2D, MsgToSpec, SpecToMsg
from torch.utils.data import DataLoader
from train import AudioWatermarkingDataset, custom_collate_fn
from eval import evaluate_model

# Load checkpoint
checkpoint_path = "./checkpoints/model_bs4_lr0.0001_ep30_epoch30.pt"  # Change this
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)

# Initialize models
inn = INN2D(in_channels=4, hidden_channels=64, n_blocks=4).to(device)
msg_encoder = MsgToSpec(msg_len=32, msg_channels=1, base_spatial=(8, 8), hidden=128).to(device)
msg_decoder = SpecToMsg(in_channels=4, msg_len=32, hidden=128).to(device)

# Load weights
inn.load_state_dict(checkpoint['inn'])
msg_encoder.load_state_dict(checkpoint['msg_encoder'])
msg_decoder.load_state_dict(checkpoint['msg_decoder'])

# Prepare dataset
audio_files = list(Path("./Dataset/dev-clean/LibriSpeech/dev-clean").rglob("*.flac"))[:100]
dataset = AudioWatermarkingDataset(audio_files, msg_length=32, cache_dir="./cache/stft")
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

# Evaluate
print("Evaluating...")
metrics = evaluate_model(inn, msg_encoder, msg_decoder, dataloader, device, cache_dir="./cache/stft")

# Print results
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Total Loss:           {metrics['loss']:.6f}")
print(f"Message Loss:         {metrics['msg_loss']:.6f}")
print(f"Reconstruction Loss:  {metrics['rec_loss']:.6f}")
print(f"Bit Accuracy:         {metrics['bit_accuracy']*100:.2f}%")
print("="*50)

# Plot accuracy
bit_accuracy = metrics['bit_accuracy']
plt.figure(figsize=(6, 5))
plt.bar(['Bit Accuracy'], [bit_accuracy * 100], color='purple', alpha=0.7, edgecolor='black', width=0.5)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Watermark Bit Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0, 105])
plt.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect (100%)')
plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
plt.text(0, bit_accuracy * 100 + 2, f'{bit_accuracy*100:.2f}%', 
         ha='center', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# ============================================
# CELL 4: Compare Multiple Checkpoints
# ============================================
import matplotlib.pyplot as plt
import torch
from pathlib import Path

checkpoint_paths = [
    "./checkpoints/model_epoch10.pt",
    "./checkpoints/model_epoch20.pt",
    "./checkpoints/model_epoch30.pt",
]

checkpoint_names = ["Epoch 10", "Epoch 20", "Epoch 30"]
losses = []
accuracies = []

for path in checkpoint_paths:
    try:
        checkpoint = torch.load(path, map_location='cpu')
        metrics = checkpoint.get('metrics', {})
        losses.append(metrics.get('loss', 0))
        # Note: bit_accuracy not in training metrics, need to evaluate separately
    except:
        losses.append(0)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.bar(checkpoint_names, losses, color=['blue', 'orange', 'green'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Loss Comparison Across Checkpoints', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (name, loss) in enumerate(zip(checkpoint_names, losses)):
    ax.text(i, loss, f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
