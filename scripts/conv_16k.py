import os
import librosa
import soundfile as sf

def resample_audio_folder(input_dir, output_dir, target_sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # Load with original sampling rate
                audio, sr = librosa.load(input_path, sr=None)

                # Resample to target_sr if needed
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                # Save as 16kHz WAV
                sf.write(output_path, audio, target_sr)
                print(f"✅ Resampled {filename} to {target_sr} Hz")

            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")

def main():
    # Update these paths
    input_root = r"C:\Users\HP\Downloads\PycharmProjects\training_haid\dataset1"
    output_root = r"C:\Users\HP\Downloads\PycharmProjects\training_haid\dataset_16k"
    subfolders = ["clean", "noise"]  # updated to match your folders

    for sub in subfolders:
        input_path = os.path.join(input_root, sub)
        output_path = os.path.join(output_root, sub)
        resample_audio_folder(input_path, output_path)

if __name__ == "__main__":
    main()
