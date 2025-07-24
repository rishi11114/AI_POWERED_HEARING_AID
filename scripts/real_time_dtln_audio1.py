#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
import tensorflow as tf
import argparse

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
    description="Real-time DTLN audio enhancement using TensorFlow Lite.",
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

# Load TFLite models
interpreter_1 = tflite.Interpreter(model_path='./pretrained_model/model_1.tflite')
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./pretrained_model/model_2.tflite')
interpreter_2.allocate_tensors()

# Get I/O details
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

# Create buffers
states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)
block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
block_len = int(np.round(fs_target * (block_len_ms / 1000)))
in_buffer = np.zeros((block_len,), dtype=np.float32)
out_buffer = np.zeros((block_len,), dtype=np.float32)

# Audio processing callback
def callback(indata, outdata, frames, time, status):
    global in_buffer, out_buffer, states_1, states_2
    if status:
        print(status)

    # Update input buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata).astype(np.float32)

    # FFT and extract mag/phase
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

    # Model 1
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.invoke()
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

    # Apply mask & inverse FFT
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype(np.float32)

    # Model 2
    interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.invoke()
    out_block = interpreter_2.get_tensor(output_details_2[0]['index'])
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])

    # Overlap-add
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = 0.0
    out_buffer += np.squeeze(out_block)

    # Output audio
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)

# Start real-time stream
try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=fs_target, blocksize=block_shift,
                   dtype=np.float32, latency=args.latency,
                   channels=1, callback=callback):
        print('#' * 80)
        print('Real-time DTLN running. Press Enter to quit.')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
