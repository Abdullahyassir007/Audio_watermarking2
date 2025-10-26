import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from models import INN2D, MsgToSpec, SpecToMsg, build_inn_input
from audiowatermarking import load_and_preprocess, wav_to_stft_tensor
import torch.nn.functional as F
import os
import argparse

class AudioWatermarkingDataset(Dataset):
    def __init__(self, audio_files, msg_length=32, cache_dir=None, transform=None):
        self.audio_files = [str(f) for f in audio_files]  # Convert Path objects to strings
        self.msg_length = msg_length
        self.cache_dir = cache_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load and preprocess audio
        audio_file = self.audio_files[idx]
        
        # Generate random message as numpy array first
        message = np.random.randint(0, 2, (self.msg_length,)).astype(np.float32)
        
        # Convert to tensor
        message = torch.from_numpy(message)
        
        # Return audio path and message (cache_dir is stored in dataset)
        return audio_file, message

def custom_collate_fn(batch):
    """Custom collate function to handle batch data"""
    audio_paths = [item[0] for item in batch]
    messages = torch.stack([item[1] for item in batch])
    return audio_paths, messages

def train_one_epoch(model, msg_encoder, msg_decoder, dataloader, optimizer, device, epoch, cache_dir):
    model.train()
    msg_encoder.train()
    msg_decoder.train()
    
    total_loss = 0.0
    total_msg_loss = 0.0
    total_rec_loss = 0.0
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch in progress:
        # Unpack the batch
        audio_paths, messages = batch
        messages = messages.to(device)
        
        # Process each audio file in the batch
        batch_loss = 0.0
        batch_msg_loss = 0.0
        batch_rec_loss = 0.0
        
        for i, audio_path in enumerate(audio_paths):
            # Load and process audio
            message = messages[i].unsqueeze(0)  # Add batch dim
            
            # Get STFT (with verbose=False and plot=False to suppress output)
            stft = wav_to_stft_tensor(
                input_data=audio_path,
                cache_dir=cache_dir,
                verbose=False,
                plot=False
            ).unsqueeze(0).to(device)  # Add batch dim
            
            # Encode message
            msg_map = msg_encoder(message, target_shape=stft.shape[2:])
            
            # Build INN input
            x = build_inn_input(
                stft_real=stft[:, 0],  # Real part
                stft_imag=stft[:, 1],  # Imaginary part
                msg_map=msg_map,
                aux_map=None
            )
            
            # Forward pass
            y, logdet = model(x)
            
            # Reconstruct
            x_recon, _ = model(y, reverse=True)
            
            # Decode message
            decoded_msg = msg_decoder(y)
            
            # Compute losses
            rec_loss = F.mse_loss(x_recon, x)
            msg_loss = F.binary_cross_entropy(decoded_msg, message)
            loss = rec_loss + msg_loss
            
            batch_loss += loss.item()
            batch_msg_loss += msg_loss.item()
            batch_rec_loss += rec_loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update progress
        avg_batch_loss = batch_loss / len(audio_paths)
        avg_batch_msg = batch_msg_loss / len(audio_paths)
        avg_batch_rec = batch_rec_loss / len(audio_paths)
        
        total_loss += batch_loss
        total_msg_loss += batch_msg_loss
        total_rec_loss += batch_rec_loss
        
        progress.set_postfix({
            'loss': avg_batch_loss,
            'msg_loss': avg_batch_msg,
            'rec_loss': avg_batch_rec
        })
    
    return {
        'loss': total_loss / len(dataloader.dataset),
        'msg_loss': total_msg_loss / len(dataloader.dataset),
        'rec_loss': total_rec_loss / len(dataloader.dataset)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Audio Watermarking Model')
    parser.add_argument('--no-cache', action='store_true', 
                        help='Disable caching (default: caching enabled)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/model_bs4_ep10.pt)')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for checkpoint files (default: auto-generated from config)')
    args = parser.parse_args()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup caching based on argument
    use_cache = not args.no_cache
    
    if use_cache:
        print("Caching enabled - processed files will be saved for faster loading")
        # Create cache directories
        os.makedirs("./cache/stft", exist_ok=True)
        os.makedirs("./cache/wavform", exist_ok=True)
        cache_dir = "./cache/stft"
    else:
        print("Caching disabled - files will be processed fresh each time")
        cache_dir = None
    
    # Hyperparameters
    batch_size = args.batch_size
    msg_length = 32
    learning_rate = args.lr
    num_epochs = args.epochs
    
    # Create checkpoint directory
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Generate checkpoint name
    if args.name:
        checkpoint_name = args.name
    else:
        checkpoint_name = f"model_bs{batch_size}_lr{learning_rate}_ep{num_epochs}"
    
    # Initialize models
    inn = INN2D(in_channels=4, hidden_channels=64, n_blocks=4).to(device)
    msg_encoder = MsgToSpec(msg_len=msg_length, msg_channels=1, base_spatial=(8, 8), hidden=128).to(device)
    msg_decoder = SpecToMsg(in_channels=4, msg_len=msg_length, hidden=128).to(device)
    
    # Optimizer
    params = list(inn.parameters()) + list(msg_encoder.parameters()) + list(msg_decoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            inn.load_state_dict(checkpoint['inn'])
            msg_encoder.load_state_dict(checkpoint['msg_encoder'])
            msg_decoder.load_state_dict(checkpoint['msg_decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting fresh training.")
    else:
        print("Starting fresh training")
    
    # Load dataset (using the same audio files as in audiowatermarking.py)
    audio_dir = Path("./Dataset/dev-clean/LibriSpeech/dev-clean")
    audio_files = list(audio_dir.rglob("*.flac"))[:100]  # Use first 100 files for testing
    
    dataset = AudioWatermarkingDataset(audio_files, msg_length=msg_length, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    # Training loop
    total_epochs = start_epoch + num_epochs
    for epoch in range(start_epoch, total_epochs):
        metrics = train_one_epoch(inn, msg_encoder, msg_decoder, dataloader, optimizer, device, epoch, cache_dir)
        
        print(f"Epoch {epoch+1}/{total_epochs}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Message Loss: {metrics['msg_loss']:.6f}")
        print(f"  Reconstruction Loss: {metrics['rec_loss']:.6f}")
        
        # Save checkpoint with custom name
        checkpoint_path = f'./checkpoints/{checkpoint_name}_epoch{epoch+1}.pt'
        torch.save({
            'inn': inn.state_dict(),
            'msg_encoder': msg_encoder.state_dict(),
            'msg_decoder': msg_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'msg_length': msg_length
            }
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    print(f"\nTraining complete! Final checkpoint: ./checkpoints/{checkpoint_name}_epoch{total_epochs}.pt")

if __name__ == "__main__":
    main()
