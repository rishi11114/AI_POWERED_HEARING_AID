import os
import librosa
import soundfile as sf
import numpy as np
import random

def mix_audio(clean_dir, noise_dir, output_clean_dir, output_noisy_dir, snr_db=5):
    """
    Mix clean audio with noise at a specified SNR and save paired files.
    """
    # Create output directories if they don't exist
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    # Get lists of WAV files
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".wav")])
    noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith(".wav")])

    # Validate input files
    if not clean_files:
        raise ValueError(f"No WAV files found in {clean_dir}")
    if not noise_files:
        raise ValueError(f"No WAV files found in {noise_dir}")

    # Process each clean file
    for i, clean_file in enumerate(clean_files):
        clean_path = os.path.join(clean_dir, clean_file)

        try:
            # Load audio files
            clean, _ = librosa.load(clean_path, sr=16000)
            # Randomly pick a noise file
            noise_file = random.choice(noise_files)
            noise_path = os.path.join(noise_dir, noise_file)
            noise, _ = librosa.load(noise_path, sr=16000)

            # Trim/pad noise to match clean length
            if len(noise) < len(clean):
                repeat_factor = int(np.ceil(len(clean) / len(noise)))
                noise = np.tile(noise, repeat_factor)
            noise = noise[:len(clean)]

            # Adjust noise to achieve target SNR
            clean_rms = np.sqrt(np.mean(clean**2))
            noise_rms = np.sqrt(np.mean(noise**2))
            if noise_rms == 0:
                raise ValueError(f"Zero RMS detected in noise file {noise_file}")
            desired_noise_rms = clean_rms / (10 ** (snr_db / 20))
            noise = noise * (desired_noise_rms / (noise_rms + 1e-8))

            # Mix audio
            noisy = clean + noise

            # Save output files
            noisy_path = os.path.join(output_noisy_dir, clean_file)
            clean_output_path = os.path.join(output_clean_dir, clean_file)
            sf.write(noisy_path, noisy, 16000)
            sf.write(clean_output_path, clean, 16000)

            print(f"✅ Mixed {clean_file} with {noise_file} → SNR: {snr_db} dB")

        except Exception as e:
            print(f"❌ Error processing {clean_file} with {noise_file}: {e}")
            continue

def main():
    base_dir = r"C:\Users\HP\Downloads\PycharmProjects\training_haid\dataset_16k"
    clean_dir = os.path.join(base_dir, "clean")
    noise_dir = os.path.join(base_dir, "noise")

    output_clean_dir = os.path.join(base_dir, "processed", "clean")
    output_noisy_dir = os.path.join(base_dir, "processed", "noisy")

    try:
        mix_audio(clean_dir, noise_dir, output_clean_dir, output_noisy_dir, snr_db=5)
    except Exception as e:
        print(f"❌ Failed to mix audio: {e}")

if __name__ == "__main__":
    main()