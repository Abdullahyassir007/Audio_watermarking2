import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import os
from models import INN2D, MsgToSpec, SpecToMsg, build_inn_input
from audiowatermarking import wav_to_stft_tensor
from train import AudioWatermarkingDataset, custom_collate_fn
from plot_utils import (plot_evaluation_results, plot_bit_comparison, 
                        plot_spectrogram_comparison, save_metrics_to_json)

def evaluate_model(model, msg_encoder, msg_decoder, dataloader, device, cache_dir=None):
    """Evaluate the model on a dataset"""
    model.eval()
    msg_encoder.eval()
    msg_decoder.eval()
    
    total_loss = 0.0
    total_msg_loss = 0.0
    total_rec_loss = 0.0
    total_bit_accuracy = 0.0
    num_samples = 0
    
    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", ncols=100)
        
        for batch in progress:
            audio_paths, messages = batch
            messages = messages.to(device)
            
            for i, audio_path in enumerate(audio_paths):
                message = messages[i].unsqueeze(0)
                
                # Get STFT
                stft = wav_to_stft_tensor(
                    input_data=audio_path,
                    cache_dir=cache_dir,
                    verbose=False,
                    plot=False
                ).unsqueeze(0).to(device)
                
                # Encode message
                msg_map = msg_encoder(message, target_shape=stft.shape[2:])
                
                # Build INN input
                x = build_inn_input(
                    stft_real=stft[:, 0],
                    stft_imag=stft[:, 1],
                    msg_map=msg_map,
                    aux_map=None
                )
                
                # Forward pass
                y, _ = model(x)
                
                # Reconstruct
                x_recon, _ = model(y, reverse=True)
                
                # Decode message
                decoded_msg = msg_decoder(y)
                
                # Compute losses
                rec_loss = F.mse_loss(x_recon, x)
                msg_loss = F.binary_cross_entropy(decoded_msg, message)
                loss = rec_loss + msg_loss
                
                # Compute bit accuracy
                predicted_bits = (decoded_msg > 0.5).float()
                bit_accuracy = (predicted_bits == message).float().mean()
                
                total_loss += loss.item()
                total_msg_loss += msg_loss.item()
                total_rec_loss += rec_loss.item()
                total_bit_accuracy += bit_accuracy.item()
                num_samples += 1
                
                progress.set_postfix({
                    'loss': total_loss / num_samples,
                    'bit_acc': total_bit_accuracy / num_samples
                })
    
    return {
        'loss': total_loss / num_samples,
        'msg_loss': total_msg_loss / num_samples,
        'rec_loss': total_rec_loss / num_samples,
        'bit_accuracy': total_bit_accuracy / num_samples
    }

def test_watermark_extraction(model, msg_encoder, msg_decoder, audio_path, device, cache_dir=None):
    """Test watermark embedding and extraction on a single audio file"""
    model.eval()
    msg_encoder.eval()
    msg_decoder.eval()
    
    # Generate random message
    msg_length = 32
    original_message = torch.randint(0, 2, (1, msg_length)).float().to(device)
    
    print(f"\nTesting on: {audio_path}")
    print(f"Original message: {original_message.cpu().numpy().astype(int).flatten()}")
    
    with torch.no_grad():
        # Get STFT
        stft = wav_to_stft_tensor(
            input_data=audio_path,
            cache_dir=cache_dir,
            verbose=False,
            plot=False
        ).unsqueeze(0).to(device)
        
        # Encode message
        msg_map = msg_encoder(original_message, target_shape=stft.shape[2:])
        
        # Build INN input
        x = build_inn_input(
            stft_real=stft[:, 0],
            stft_imag=stft[:, 1],
            msg_map=msg_map,
            aux_map=None
        )
        
        # Forward pass (embed watermark)
        y, _ = model(x)
        
        # Decode message (extract watermark)
        decoded_msg = msg_decoder(y)
        predicted_bits = (decoded_msg > 0.5).float()
        
        # Compute metrics
        bit_accuracy = (predicted_bits == original_message).float().mean().item()
        msg_loss = F.binary_cross_entropy(decoded_msg, original_message).item()
        
        print(f"Decoded message:  {predicted_bits.cpu().numpy().astype(int).flatten()}")
        print(f"Bit Accuracy: {bit_accuracy * 100:.2f}%")
        print(f"Message Loss: {msg_loss:.6f}")
        
        # Show bit-by-bit comparison
        original_bits = original_message.cpu().numpy().astype(int).flatten()
        decoded_bits = predicted_bits.cpu().numpy().astype(int).flatten()
        errors = np.where(original_bits != decoded_bits)[0]
        
        if len(errors) > 0:
            print(f"Errors at positions: {errors.tolist()}")
        else:
            print("Perfect reconstruction!")
    
    return bit_accuracy, msg_loss

def main():
    parser = argparse.ArgumentParser(description='Evaluate Audio Watermarking Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'single'],
                        help='Evaluation mode: full (entire dataset) or single (one file)')
    parser.add_argument('--audio-file', type=str, default=None,
                        help='Path to audio file for single mode')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--num-files', type=int, default=100,
                        help='Number of files to evaluate (default: 100)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup caching
    cache_dir = None if args.no_cache else "./cache/stft"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs("./cache/wavform", exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize models
    msg_length = checkpoint.get('config', {}).get('msg_length', 32)
    inn = INN2D(in_channels=4, hidden_channels=64, n_blocks=4).to(device)
    msg_encoder = MsgToSpec(msg_len=msg_length, msg_channels=1, base_spatial=(8, 8), hidden=128).to(device)
    msg_decoder = SpecToMsg(in_channels=4, msg_len=msg_length, hidden=128).to(device)
    
    # Load weights
    inn.load_state_dict(checkpoint['inn'])
    msg_encoder.load_state_dict(checkpoint['msg_encoder'])
    msg_decoder.load_state_dict(checkpoint['msg_decoder'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    if 'metrics' in checkpoint:
        print(f"Training metrics: {checkpoint['metrics']}")
    
    if args.mode == 'single':
        # Single file evaluation
        if not args.audio_file:
            print("Error: --audio-file required for single mode")
            return
        
        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file not found: {args.audio_file}")
            return
        
        test_watermark_extraction(inn, msg_encoder, msg_decoder, args.audio_file, device, cache_dir)
    
    else:
        # Full dataset evaluation
        print(f"\nEvaluating on dataset...")
        audio_dir = Path("./Dataset/dev-clean/LibriSpeech/dev-clean")
        audio_files = list(audio_dir.rglob("*.flac"))[:args.num_files]
        
        if len(audio_files) == 0:
            print("Error: No audio files found in dataset")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        dataset = AudioWatermarkingDataset(audio_files, msg_length=msg_length, cache_dir=cache_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=2, collate_fn=custom_collate_fn)
        
        metrics = evaluate_model(inn, msg_encoder, msg_decoder, dataloader, device, cache_dir)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total Loss:           {metrics['loss']:.6f}")
        print(f"Message Loss:         {metrics['msg_loss']:.6f}")
        print(f"Reconstruction Loss:  {metrics['rec_loss']:.6f}")
        print(f"Bit Accuracy:         {metrics['bit_accuracy']*100:.2f}%")
        print("="*50)
        
        # Plot accuracy
        from plot_utils import plot_accuracy
        plot_accuracy(metrics['bit_accuracy'])
        
        # Test on a few random samples
        print("\nTesting on random samples:")
        random_indices = np.random.choice(len(audio_files), min(3, len(audio_files)), replace=False)
        for idx in random_indices:
            test_watermark_extraction(inn, msg_encoder, msg_decoder, 
                                     str(audio_files[idx]), device, cache_dir)

if __name__ == "__main__":
    main()
