# Google Colab Usage Guide

## Quick Start - Copy & Paste These Cells

### Cell 1: Plot Training History
```python
import matplotlib.pyplot as plt
import torch
from pathlib import Path

checkpoint_dir = "./checkpoints"
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
    
    print(f"Final Loss: {losses[-1]:.4f} | Best Loss: {min(losses):.4f}")
```

### Cell 2: Plot Accuracy After Evaluation
```python
import matplotlib.pyplot as plt

# Get bit_accuracy from your evaluation results
bit_accuracy = 0.98  # Replace with actual value

plt.figure(figsize=(6, 5))
plt.bar(['Bit Accuracy'], [bit_accuracy * 100], color='purple', alpha=0.7, edgecolor='black', width=0.5)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Watermark Bit Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0, 105])
plt.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect')
plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random')
plt.text(0, bit_accuracy * 100 + 2, f'{bit_accuracy*100:.2f}%', 
         ha='center', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Cell 3: Full Evaluation with Auto-Plot
```python
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from models import INN2D, MsgToSpec, SpecToMsg
from torch.utils.data import DataLoader
from train import AudioWatermarkingDataset, custom_collate_fn
from eval import evaluate_model

# Configuration
checkpoint_path = "./checkpoints/model_epoch30.pt"  # Change this
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
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
metrics = evaluate_model(inn, msg_encoder, msg_decoder, dataloader, device, cache_dir="./cache/stft")

# Print results
print(f"\nBit Accuracy: {metrics['bit_accuracy']*100:.2f}%")
print(f"Total Loss: {metrics['loss']:.6f}")

# Plot
bit_accuracy = metrics['bit_accuracy']
plt.figure(figsize=(6, 5))
plt.bar(['Bit Accuracy'], [bit_accuracy * 100], color='purple', alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy (%)')
plt.title('Watermark Bit Accuracy')
plt.ylim([0, 105])
plt.axhline(y=100, color='green', linestyle='--', label='Perfect')
plt.axhline(y=50, color='red', linestyle='--', label='Random')
plt.text(0, bit_accuracy * 100 + 2, f'{bit_accuracy*100:.2f}%', ha='center', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

## Training in Colab

```python
# Train with default settings
!python train.py --epochs 30

# Train without cache
!python train.py --no-cache --epochs 30 --batch-size 8

# Resume training
!python train.py --resume checkpoints/model_epoch30.pt --epochs 20
```

## Evaluation in Colab

```python
# Full evaluation
!python eval.py --checkpoint checkpoints/model_epoch30.pt

# Single file test
!python eval.py --checkpoint checkpoints/model_epoch30.pt \
                --mode single \
                --audio-file Dataset/dev-clean/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac
```

## Tips for Colab

1. **Mount Google Drive** to save checkpoints:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Check GPU availability**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

3. **Monitor training** in real-time - the progress bar will show in Colab output

4. **Save plots** to Drive:
```python
plt.savefig('/content/drive/MyDrive/training_plot.png', dpi=150, bbox_inches='tight')
```
