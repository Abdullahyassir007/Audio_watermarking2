# Audio Watermarking Evaluation Guide

## Quick Start

### Evaluate on Full Dataset
```bash
# Evaluate a trained model on 100 audio files
python eval.py --checkpoint checkpoints/model_bs4_lr0.0001_ep30_epoch30.pt

# Evaluate on more files
python eval.py --checkpoint checkpoints/model_bs4_lr0.0001_ep30_epoch30.pt --num-files 500

# Evaluate without caching
python eval.py --checkpoint checkpoints/model_bs4_lr0.0001_ep30_epoch30.pt --no-cache
```

### Test on Single Audio File
```bash
# Test watermark embedding and extraction on a specific file
python eval.py --checkpoint checkpoints/model_bs4_lr0.0001_ep30_epoch30.pt \
               --mode single \
               --audio-file Dataset/dev-clean/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac
```

## Evaluation Modes

### 1. Full Dataset Evaluation (`--mode full`)
Evaluates the model on multiple audio files and provides aggregate metrics:
- **Total Loss**: Combined reconstruction and message loss
- **Message Loss**: Binary cross-entropy loss for message decoding
- **Reconstruction Loss**: MSE loss for audio reconstruction
- **Bit Accuracy**: Percentage of correctly decoded message bits

**Example Output:**
```
==================================================
EVALUATION RESULTS
==================================================
Total Loss:           0.234567
Message Loss:         0.123456
Reconstruction Loss:  0.000012
Bit Accuracy:         98.75%
==================================================
```

### 2. Single File Testing (`--mode single`)
Tests watermark embedding and extraction on one audio file:
- Shows original vs decoded message bits
- Displays bit-by-bit accuracy
- Identifies error positions

**Example Output:**
```
Testing on: Dataset/dev-clean/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac
Original message: [1 0 1 1 0 0 1 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 1 1 0 0 1 0 1 1]
Decoded message:  [1 0 1 1 0 0 1 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 1 1 0 0 1 0 1 1]
Bit Accuracy: 100.00%
Message Loss: 0.000123
Perfect reconstruction!
```

## Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--checkpoint` | str | Yes | - | Path to checkpoint file to evaluate |
| `--mode` | str | No | `full` | Evaluation mode: `full` or `single` |
| `--audio-file` | str | No | None | Audio file path (required for `single` mode) |
| `--batch-size` | int | No | 4 | Batch size for evaluation |
| `--no-cache` | flag | No | False | Disable caching of processed audio |
| `--num-files` | int | No | 100 | Number of files to evaluate in full mode |

## Usage Examples

### Example 1: Quick evaluation after training
```bash
# Train a model
python train.py --epochs 30 --name my_model

# Evaluate it
python eval.py --checkpoint checkpoints/my_model_epoch30.pt
```

### Example 2: Compare different checkpoints
```bash
# Evaluate early checkpoint
python eval.py --checkpoint checkpoints/my_model_epoch10.pt --num-files 200

# Evaluate final checkpoint
python eval.py --checkpoint checkpoints/my_model_epoch50.pt --num-files 200
```

### Example 3: Test on specific audio files
```bash
# Test on a clean speech file
python eval.py --checkpoint checkpoints/my_model_epoch30.pt \
               --mode single \
               --audio-file Dataset/dev-clean/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac

# Test on another file
python eval.py --checkpoint checkpoints/my_model_epoch30.pt \
               --mode single \
               --audio-file Dataset/dev-clean/LibriSpeech/dev-clean/1673/143397/1673-143397-0001.flac
```

### Example 4: Large-scale evaluation
```bash
# Evaluate on entire dataset
python eval.py --checkpoint checkpoints/production_model_epoch100.pt --num-files 2707
```

### Example 5: Evaluation without cache (fresh processing)
```bash
python eval.py --checkpoint checkpoints/my_model_epoch30.pt --no-cache --num-files 50
```

## Understanding the Metrics

### Bit Accuracy
- **100%**: Perfect watermark extraction (all bits correct)
- **90-99%**: Good performance, minor errors
- **50-89%**: Poor performance, significant errors
- **~50%**: Random guessing (model not learning)

### Message Loss
- **< 0.1**: Excellent message encoding/decoding
- **0.1 - 0.3**: Good performance
- **> 0.5**: Poor performance

### Reconstruction Loss
- **< 0.001**: Excellent audio reconstruction
- **0.001 - 0.01**: Good reconstruction
- **> 0.01**: Noticeable distortion

## Colab Usage

### In Google Colab:
```python
# Mount Google Drive (if checkpoint is there)
from google.colab import drive
drive.mount('/content/drive')

# Run evaluation
!python eval.py --checkpoint /content/drive/MyDrive/checkpoints/model_epoch30.pt

# Test on single file
!python eval.py --checkpoint /content/drive/MyDrive/checkpoints/model_epoch30.pt \
                --mode single \
                --audio-file Dataset/dev-clean/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac
```

### Programmatic Evaluation in Colab:
```python
import torch
from eval import evaluate_model, test_watermark_extraction
from models import INN2D, MsgToSpec, SpecToMsg
from torch.utils.data import DataLoader
from train import AudioWatermarkingDataset, custom_collate_fn
from pathlib import Path

# Load checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/my_model_epoch30.pt', map_location=device)

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
print(f"Bit Accuracy: {metrics['bit_accuracy']*100:.2f}%")
```

## Tips

1. **Use caching** for faster evaluation on the same dataset
2. **Test on single files** first to verify model behavior
3. **Compare checkpoints** from different epochs to track improvement
4. **Evaluate on unseen data** to test generalization
5. **Check bit accuracy** as the primary metric for watermark quality
6. **Monitor reconstruction loss** to ensure audio quality is preserved
