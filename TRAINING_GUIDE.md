# Audio Watermarking Training Guide

## Quick Start

### Start Fresh Training
```bash
# Basic training (default: batch_size=4, epochs=1, lr=1e-4)
python train.py

# Custom configuration
python train.py --batch-size 8 --epochs 30 --lr 0.0001

# Without caching
python train.py --no-cache --epochs 10
```

### Resume Training from Checkpoint
```bash
# Resume from a specific checkpoint
python train.py --resume checkpoints/model_bs4_lr0.0001_ep30_epoch30.pt --epochs 20

# This will continue training for 20 MORE epochs (epochs 31-50)
```

### Custom Checkpoint Names
```bash
# Use a custom name for your checkpoints
python train.py --name my_experiment_v1 --epochs 50

# Checkpoints will be saved as:
# checkpoints/my_experiment_v1_epoch1.pt
# checkpoints/my_experiment_v1_epoch2.pt
# ...
```

## Training Scenarios

### Scenario 1: Train a new model with different hyperparameters
Each configuration will create separate checkpoint files:

```bash
# First experiment: small batch, many epochs
python train.py --batch-size 4 --epochs 50 --lr 0.0001
# Saves: checkpoints/model_bs4_lr0.0001_ep50_epoch*.pt

# Second experiment: large batch, fewer epochs
python train.py --batch-size 16 --epochs 20 --lr 0.001
# Saves: checkpoints/model_bs16_lr0.001_ep20_epoch*.pt
```

### Scenario 2: Continue training an existing model
```bash
# Train for 30 epochs
python train.py --epochs 30 --name baseline_model

# Later, continue for 20 more epochs (total: 50)
python train.py --resume checkpoints/baseline_model_epoch30.pt --epochs 20 --name baseline_model
# Will train epochs 31-50
```

### Scenario 3: Fine-tune with different learning rate
```bash
# Initial training
python train.py --epochs 30 --lr 0.001 --name model_v1

# Fine-tune with lower learning rate
python train.py --resume checkpoints/model_v1_epoch30.pt --epochs 20 --lr 0.0001 --name model_v1_finetune
```

## Checkpoint Management

### Checkpoint Structure
Each checkpoint contains:
- Model weights (INN, message encoder, message decoder)
- Optimizer state
- Current epoch number
- Training metrics
- Configuration (batch_size, learning_rate, msg_length)

### Checkpoint Location
All checkpoints are saved in the `./checkpoints/` directory.

### Checkpoint Naming
- **Auto-generated**: `model_bs{batch_size}_lr{learning_rate}_ep{epochs}_epoch{current_epoch}.pt`
- **Custom**: `{your_name}_epoch{current_epoch}.pt`

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 4 | Batch size for training |
| `--epochs` | int | 1 | Number of epochs to train |
| `--lr` | float | 1e-4 | Learning rate |
| `--no-cache` | flag | False | Disable caching of processed audio |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--name` | str | None | Custom name for checkpoint files |

## Examples

### Example 1: Quick test run
```bash
python train.py --epochs 5 --batch-size 2
```

### Example 2: Full training run
```bash
python train.py --epochs 100 --batch-size 8 --lr 0.0001 --name production_model
```

### Example 3: Resume interrupted training
```bash
# Training was interrupted at epoch 45
python train.py --resume checkpoints/production_model_epoch45.pt --epochs 55 --name production_model
# Will continue from epoch 46 to 100
```

### Example 4: Multiple experiments
```bash
# Experiment A
python train.py --name exp_a --batch-size 4 --epochs 30

# Experiment B (different config, separate checkpoints)
python train.py --name exp_b --batch-size 8 --epochs 30

# Experiment C (no caching for comparison)
python train.py --name exp_c --no-cache --batch-size 4 --epochs 30
```

## Tips

1. **Always use `--name`** for important experiments to avoid confusion
2. **Resume with same name** to keep checkpoint files organized
3. **Check checkpoint directory** regularly to manage disk space
4. **Save final checkpoints** separately before starting new experiments
5. **Use `--no-cache`** when testing to ensure fresh processing
