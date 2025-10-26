
from pathlib import Path
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import hashlib
# Path to your local dataset

DATASET_PATH = r"Dataset/dev-clean/LibriSpeech/dev-clean"

def list_audio_files(base_path):
    """
    List all FLAC audio files in the dataset directory.

    Args:
        base_path (str): Path to the dataset directory

    Returns:
        list: Sorted list of full paths to FLAC files
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}")
        return []

    # Find all .flac files recursively
    flac_files = list(base_path.rglob('*.flac'))

    # Convert to strings and sort for consistent ordering
    return sorted(str(f) for f in flac_files)

# Get audio files (only when running as main script)
audio_files = []


#filenamecreation
def filenamecreate(path,save_dir):
        path_obj = Path(path)
        # Get speaker ID from path
        parts = path_obj.parts
        speaker_id = parts[-3] if len(parts) >= 3 else 'unknown'
        # Create unique cache filename with speaker ID
        cache_filename = f"{speaker_id}_{path_obj.stem}"
        cache_path = Path(save_dir) / cache_filename

        return cache_path
      
target_sr = 16000
segment_length = target_sr * 4   # 4 seconds

resampler = T.Resample(orig_freq=48000, new_freq=target_sr)

def load_and_preprocess(path, save_dir=None, force_reload=False, verbose=False):
    """
    Load and preprocess audio file, optionally save processed chunks

    Args:
        path: Path to audio file
        save_dir: If provided, save processed chunks here
        force_reload: If False, load from cache if available
        verbose: If True, print debug information
    """
    from pathlib import Path

    # If save_dir is provided, check cache first
    if save_dir and not force_reload:
        cache_path = Path(f"{filenamecreate(path,save_dir)}.pt")
        if verbose:
            print(cache_path)

        # Try to load from cache
        try:
            if cache_path.exists():
                return torch.load(cache_path).float()  # Convert back to float32
        except Exception as e:
            if verbose:
                print(f"Error loading cached file {cache_path}: {e}")

    # Load and preprocess the audio
    try:
        wav, sr = torchaudio.load(path)

        # Resample if needed
        if sr != target_sr:
            wav = resampler(wav)

        # Convert to mono
        wav = wav.mean(dim=0)

        # Normalize amplitude
        wav = wav / wav.abs().max()

        # Pad or trim to fixed length
        if wav.shape[0] < segment_length:
            wav = torch.nn.functional.pad(wav, (0, segment_length - wav.shape[0]))
        else:
            wav = wav[:segment_length]

        # Save to cache if requested
        if save_dir:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(wav.half(), cache_path)  # Save as float16 to save space

        return wav

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def plot_spectrogram(stft_tensor,sample_wav, sample_rate=16000, hop_length=256):


    magnitude = torch.sqrt(stft_tensor[0]**2 + stft_tensor[1]**2)


    log_spectrogram = 20 * torch.log10(torch.clamp(magnitude, min=1e-10))


    times = torch.linspace(0, log_spectrogram.shape[1] * hop_length / sample_rate,
                          log_spectrogram.shape[1])
    freqs = torch.linspace(0, sample_rate/2, log_spectrogram.shape[0])

    # Create figure
    plt.figure(figsize=(15, 5))

    # Plot waveform
    plt.subplot(1, 2, 1)
    plt.plot(times, sample_wav.numpy()[:len(times)], linewidth=0.5)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Plot spectrogram
    plt.subplot(1, 2, 2)
    plt.pcolormesh(times, freqs, log_spectrogram, shading='gouraud')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Magnitude (dB)')

    plt.tight_layout()
    plt.show()

# Visualize the spectrogram

# STFT parameters
n_fft = 1024
hop_length = 256
win_length = 1024


def wav_to_stft_tensor(input_data, cache_dir=None, n_fft=1024, hop_length=256, win_length=1024, verbose=False, plot=False):
    """Convert a waveform file to STFT tensor with caching.

    Args:
        input_data: Path to audio file
        cache_dir: Directory to save/load cached STFT tensors (None to disable)
        n_fft: FFT window size
        hop_length: Number of samples between frames
        win_length: Each frame of audio is windowed by window of length win_length
        verbose: If True, print debug information
        plot: If True, plot spectrogram

    Returns:
        torch.Tensor: Complex STFT tensor of shape (2, n_freq_bins, n_frames)
    """

    # Check STFT cache first
    wav = load_and_preprocess(input_data, save_dir="./cache/wavform" if cache_dir else None, verbose=verbose)
    if cache_dir:
        cache_path =  Path(f"{filenamecreate(input_data,cache_dir)}_stft.pt")

        if verbose:
            print("Loading from cache:", cache_path)
        if cache_path.exists():
            try:
                stft_tensor = torch.load(cache_path)
                if plot:
                    plot_spectrogram(stft_tensor, wav, sample_rate=target_sr, hop_length=hop_length)
                return stft_tensor
            except Exception as e:
                if verbose:
                    print(f"Error loading cached STFT: {e}")

    # If not in STFT cache, load and preprocess
    if wav is None:
        wav = load_and_preprocess(input_data, save_dir="./cache/wavform" if cache_dir else None, verbose=verbose)
    if wav is None:
        raise ValueError(f"Failed to load audio from {input_data}")

    # Compute STFT with window function
    window = torch.hann_window(win_length, device=wav.device)
    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    # Separate real and imag
    stft_real = stft.real
    stft_imag = stft.imag

    # Stack into [channels, freq_bins, frames]
    stft_tensor = torch.stack([stft_real, stft_imag], dim=0)

    # Save to cache if requested
    if cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stft_tensor, cache_path)
        if verbose:
            print(f"Saved STFT to {cache_path}")
        if plot:
            plot_spectrogram(stft_tensor, wav, sample_rate=target_sr, hop_length=hop_length)

    return stft_tensor


if __name__ == "__main__":
    # Get and list all audio files
    audio_files = list_audio_files(DATASET_PATH)

    # Print summary
    print(f"Found {len(audio_files)} audio files")
    if audio_files:
        print("\nFirst 5 files:")
        for i, file in enumerate(audio_files[:5], 1):
            print(f"{i}. {file}")

        print("\nSpeakers and number of files:")
        # Show unique speaker IDs and number of files per speaker
        speakers = {}
        for file in audio_files:
            parts = Path(file).parts
            # Look for speaker ID in the path (it's the directory before the last one)

            speaker_id = None
            for i, part in enumerate(parts):
                if part == 'dev-clean' and i + 1 < len(parts) and parts[i+1].isdigit():
                    speaker_id = parts[i+1]
                    break

            if speaker_id is None and len(parts) > 1:
                # If we couldn't find a speaker ID, use the parent directory name as fallback
                speaker_id = parts[-2] if parts[-2] != 'dev-clean' else 'unknown'

            speakers[speaker_id] = speakers.get(speaker_id, 0) + 1

        print("\nSpeakers and number of files:")
        for speaker, count in sorted(speakers.items()):
            print(f"- Speaker {speaker}: {count} files")

        print("\nSpeaker ID mapping:", speakers)

    # Example usage
    sample_stft = wav_to_stft_tensor(
        input_data=audio_files[0],
        cache_dir="./cache/stft"
    )
    print("STFT tensor shape:", sample_stft.shape)  # (2, freq_bins, frames)