import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import tensorflow.lite as tflite  # works with TF 2.10+
import os

# Paths
input_audio_path = "input_audio/noisy_sample1.wav"
model_1_path = "pretrained_model/model_1.tflite"
model_2_path = "pretrained_model/model_2.tflite"
output_audio_path = "output_audio/enhanced_output.wav"

# Load and resample if needed
y, fs = librosa.load(input_audio_path, sr=16000)  # will auto-resample to 16k
print(f"[INFO] Audio loaded. Length: {len(y)/fs:.2f}s, Sample rate: {fs} Hz")

# Define block sizes
block_len = int(0.032 * fs)   # 32 ms
block_shift = int(0.008 * fs) # 8 ms

# Padding
num_blocks = int(np.ceil(len(y) / block_shift))
pad_len = (num_blocks - 1) * block_shift + block_len
y_pad = np.concatenate([y, np.zeros(pad_len - len(y))])
enhanced_audio = np.zeros_like(y_pad)

# Load models
interpreter_1 = tflite.Interpreter(model_path=model_1_path)
interpreter_1.allocate_tensors()
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)

interpreter_2 = tflite.Interpreter(model_path=model_2_path)
interpreter_2.allocate_tensors()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)

# Process audio block by block
for idx in range(num_blocks):
    in_block = y_pad[idx * block_shift : idx * block_shift + block_len]
    in_block_fft = np.fft.rfft(in_block)
    in_mag = np.abs(in_block_fft).astype(np.float32)
    in_phase = np.angle(in_block_fft)

    in_mag_reshaped = in_mag.reshape(1, 1, -1)

    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag_reshaped)
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.invoke()

    mask = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

    est_complex = in_mag * mask.squeeze() * np.exp(1j * in_phase)
    est_block = np.fft.irfft(est_complex).astype(np.float32)
    est_block_reshaped = est_block.reshape(1, 1, -1)

    interpreter_2.set_tensor(input_details_2[0]['index'], est_block_reshaped)
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.invoke()

    out_block = interpreter_2.get_tensor(output_details_2[0]['index']).squeeze()
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])

    enhanced_audio[idx * block_shift : idx * block_shift + block_len] += out_block

# Trim to original length
enhanced_audio = enhanced_audio[:len(y)]

# Save output
sf.write(output_audio_path, enhanced_audio, fs)
print(f"[INFO] Enhanced audio saved to: {output_audio_path}")

# Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Noisy Audio - Waveform")
librosa.display.waveshow(y, sr=fs)
plt.xlabel("Time (s)")

plt.subplot(2, 2, 2)
plt.title("Enhanced Audio - Waveform")
librosa.display.waveshow(enhanced_audio, sr=fs)
plt.xlabel("Time (s)")

plt.subplot(2, 2, 3)
plt.title("Noisy Audio - Spectrogram")
D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D1, sr=fs, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 2, 4)
plt.title("Enhanced Audio - Spectrogram")
D2 = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_audio)), ref=np.max)
librosa.display.specshow(D2, sr=fs, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()
