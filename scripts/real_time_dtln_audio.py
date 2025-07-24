import numpy as np
import sounddevice as sd
import tensorflow as tf
import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')  # Avoid GUI-related errors
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io.wavfile import write as wavwrite

# Alias tflite interpreter
tflite = tf.lite


def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

# Argument parsing
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description="Real-time DTLN audio enhancement with plotting and saving.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser]
)
parser.add_argument('-i', '--input-device', type=int_or_str, help='input device (numeric ID or substring)')
parser.add_argument('-o', '--output-device', type=int_or_str, help='output device (numeric ID or substring)')
parser.add_argument('--latency', type=float, help='latency in seconds', default=0.2)
args = parser.parse_args(remaining)

# Set parameters
block_len_ms = 32
block_shift_ms = 8
fs_target = 16000
plot_interval_sec = 3
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# Load models
interpreter_1 = tflite.Interpreter(model_path='./pretrained_model/model_1.tflite')
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./pretrained_model/model_2.tflite')
interpreter_2.allocate_tensors()

# Tensor info
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

# Buffers
states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)
block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
block_len = int(np.round(fs_target * (block_len_ms / 1000)))
in_buffer = np.zeros((block_len,), dtype=np.float32)
out_buffer = np.zeros((block_len,), dtype=np.float32)
in_accum = []
out_accum = []
last_plot_time = time.time()
plot_count = 1


def save_plot_waveform_spec(in_audio, out_audio, fs, count):
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    t = np.linspace(0, len(in_audio) / fs, len(in_audio))
    axs[0, 0].plot(t, in_audio)
    axs[0, 0].set_title("Noisy Audio - Waveform")
    axs[0, 1].plot(t, out_audio)
    axs[0, 1].set_title("Enhanced Audio - Waveform")

    f1, t1, Sxx1 = spectrogram(in_audio, fs)
    f2, t2, Sxx2 = spectrogram(out_audio, fs)
    axs[1, 0].pcolormesh(t1, f1, 10 * np.log10(Sxx1 + 1e-10))
    axs[1, 0].set_title("Noisy Audio - Spectrogram")
    axs[1, 1].pcolormesh(t2, f2, 10 * np.log10(Sxx2 + 1e-10))
    axs[1, 1].set_title("Enhanced Audio - Spectrogram")

    for ax in axs.flat:
        ax.set_xlabel("Time")
        ax.set_ylabel("Hz")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_{count}.png'))
    plt.close()


def callback(indata, outdata, frames, time_data, status):
    global in_buffer, out_buffer, states_1, states_2, in_accum, out_accum, last_plot_time, plot_count

    if status:
        print(status)

    # Convert input to float32 and resample if needed
    input_chunk = np.squeeze(indata).astype(np.float32)
    if indata.shape[0] != block_shift:
        return

    # Update buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = input_chunk

    # FFT
    in_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_fft)
    in_phase = np.angle(in_fft)
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

    # Model 1
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.invoke()
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

    # IFFT
    est_complex = in_mag * out_mask * np.exp(1j * in_phase)
    est_block = np.fft.irfft(est_complex)
    est_block = np.reshape(est_block, (1, 1, -1)).astype(np.float32)

    # Model 2
    interpreter_2.set_tensor(input_details_2[0]['index'], est_block)
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.invoke()
    out_block = interpreter_2.get_tensor(output_details_2[0]['index'])
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])

    # Overlap-add
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = 0.0
    out_buffer += np.squeeze(out_block)

    # Output
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)

    # Store for plotting
    in_accum.extend(input_chunk.tolist())
    out_accum.extend(out_buffer[:block_shift].tolist())

    # Save plots periodically
    if time.time() - last_plot_time >= plot_interval_sec:
        save_plot_waveform_spec(np.array(in_accum), np.array(out_accum), fs_target, plot_count)
        in_accum = []
        out_accum = []
        last_plot_time = time.time()
        plot_count += 1


# Run stream
try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=fs_target, blocksize=block_shift,
                   dtype=np.float32, latency=args.latency,
                   channels=1, callback=callback):
        print("#" * 80)
        print("Real-time DTLN running with visualization. Press Enter to quit.")
        print("#" * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
