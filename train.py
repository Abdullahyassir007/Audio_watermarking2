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

class AudioWatermarkingDataset(Dataset):
    def __init__(self, audio_files, msg_length=32, transform=None):
        self.audio_files = [str(f) for f in audio_files]  # Convert Path objects to strings
        self.msg_length = msg_length
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
        
        # Return as a tuple of tensors (required for default_collate)
        return audio_file, message

def train_one_epoch(model, msg_encoder, msg_decoder, dataloader, optimizer, device, epoch):
    model.train()
    msg_encoder.train()
    msg_decoder.train()
    
    total_loss = 0.0
    total_msg_loss = 0.0
    total_rec_loss = 0.0
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
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
            
            # Get STFT
            stft = wav_to_stft_tensor(
                input_data=audio_path,
                cache_dir="./cache/stft"
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
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4
    msg_length = 32
    learning_rate = 1e-4
    num_epochs = 1
    
    # Create cache directories
    os.makedirs("./cache/stft", exist_ok=True)
    os.makedirs("./cache/wavform", exist_ok=True)
    
    # Initialize models
    inn = INN2D(in_channels=4, hidden_channels=64, n_blocks=4).to(device)
    msg_encoder = MsgToSpec(msg_len=msg_length, msg_channels=1, base_spatial=(8, 8), hidden=128).to(device)
    msg_decoder = SpecToMsg(in_channels=4, msg_len=msg_length, hidden=128).to(device)
    
    # Optimizer
    params = list(inn.parameters()) + list(msg_encoder.parameters()) + list(msg_decoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    
    # Load dataset (using the same audio files as in audiowatermarking.py)
    audio_dir = Path("./Dataset/dev-clean/LibriSpeech/dev-clean")
    audio_files = list(audio_dir.rglob("*.flac"))[:100]  # Use first 100 files for testing
    
    dataset = AudioWatermarkingDataset(audio_files, msg_length=msg_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Training loop
    for epoch in range(num_epochs):
        metrics = train_one_epoch(inn, msg_encoder, msg_decoder, dataloader, optimizer, device, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Message Loss: {metrics['msg_loss']:.6f}")
        print(f"  Reconstruction Loss: {metrics['rec_loss']:.6f}")
        
        # Save models
        torch.save({
            'inn': inn.state_dict(),
            'msg_encoder': msg_encoder.state_dict(),
            'msg_decoder': msg_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }, f'checkpoint_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    main()
