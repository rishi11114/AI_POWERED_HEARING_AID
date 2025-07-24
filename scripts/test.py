import argparse
import sys
import signal
import threading

import numpy as np
import sounddevice as sd
import tensorflow as tf
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

_base = argparse.ArgumentParser(add_help=False)
_base.add_argument('-l', '--list-devices', action='store_true',
                   help='show list of audio devices and exit')
_args_tmp, _remaining = _base.parse_known_args()
if _args_tmp.list_devices:
    print(sd.query_devices())
    sys.exit(0)

parser = argparse.ArgumentParser(
    description="Real-time DTLN (TFLite) with PyQtGraph live plots + diff heatmap.",
    parents=[_base],
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('-i', '--input-device', type=int_or_str, help='input device id or name')
parser.add_argument('-o', '--output-device', type=int_or_str, help='output device id or name')
parser.add_argument('--latency', type=float, default=0.25, help='audio latency hint (sec)')
parser.add_argument('--plot-window', type=float, default=3.0,
                    help='seconds of audio history shown in plots')
parser.add_argument('--plot-interval', type=float, default=0.05,
                    help='GUI update period (sec)')
parser.add_argument('--db-min', type=float, default=-80.0, help='spectrogram dB floor (ignored if --auto-contrast)')
parser.add_argument('--db-max', type=float, default=0.0, help='spectrogram dB ceiling (ignored if --auto-contrast)')
parser.add_argument('--auto-contrast', action='store_true',
                    help='auto-scale spectrogram dB levels from live signal (percentile clipped)')
parser.add_argument('--cmap', type=str, default='inferno',
                    help='colormap name (inferno, viridis, plasma, magma, etc.)')
parser.add_argument('--no-grid', action='store_true',
                    help='disable grid lines on spectrogram axes')
parser.add_argument('--diff-max', type=float, default=40.0,
                    help='max dB shown in spectral difference heatmap')
parser.add_argument('--model-dir', type=str, default='./pretrained_model',
                    help='dir with model_1.tflite & model_2.tflite')
cli = parser.parse_args(_remaining)


# ------------------------------------------------------------------
# Core params
# ------------------------------------------------------------------
fs_target = 16000
block_len_ms = 32     # 32 ms frame
block_shift_ms = 8    # 8 ms hop

block_shift = int(round(fs_target * (block_shift_ms / 1000.0)))  # 128
block_len   = int(round(fs_target * (block_len_ms   / 1000.0)))  # 512

# ------------------------------------------------------------------
# Load TFLite models
# ------------------------------------------------------------------
tflite = tf.lite
model_1_path = f"{cli.model_dir}/model_1.tflite"
model_2_path = f"{cli.model_dir}/model_2.tflite"

interpreter_1 = tflite.Interpreter(model_path=model_1_path)
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path=model_2_path)
interpreter_2.allocate_tensors()

input_details_1  = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2  = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

# LSTM state buffers
states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)

# Processing buffers (rolling 32ms frame)
in_buffer  = np.zeros(block_len, dtype=np.float32)
out_buffer = np.zeros(block_len, dtype=np.float32)

# ------------------------------------------------------------------
# Visualization buffers
# ------------------------------------------------------------------
wave_hist_len = int(cli.plot_window * fs_target)      # samples in waveform history
spec_cols     = max(1, wave_hist_len // block_shift)  # columns in spectrogram history
spec_rows     = block_len // 2 + 1                    # rfft bins

wave_in_hist  = np.zeros(wave_hist_len, dtype=np.float32)
wave_out_hist = np.zeros(wave_hist_len, dtype=np.float32)
spec_in_hist  = np.zeros((spec_rows, spec_cols), dtype=np.float32)
spec_out_hist = np.zeros((spec_rows, spec_cols), dtype=np.float32)

hist_lock = threading.Lock()

# Time axis for waveforms (negative to 0 sec)
wave_time = np.linspace(-cli.plot_window, 0, wave_hist_len, endpoint=False)

# Fixed dB display range (if auto-contrast off)
db_min = float(cli.db_min)
db_max = float(cli.db_max)
db_range = (db_min, db_max)
diff_max = float(cli.diff_max)  # max dB of reduction shown


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def mag2db(x):
    """Magnitude -> dB."""
    return 20.0 * np.log10(np.maximum(x, 1e-10))

def make_lut(name='inferno', n=256):
    """Return RGBA lookup table for PyQtGraph from a matplotlib colormap name."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name, n)
        lut = (cmap(np.linspace(0.0, 1.0, n)) * 255).astype(np.ubyte)
        return lut
    except Exception:
        # fallback grayscale
        arr = np.linspace(0, 255, n, dtype=np.ubyte)
        return np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=1)

lut = make_lut(cli.cmap)


# ------------------------------------------------------------------
# PyQtGraph GUI (main thread)
# ------------------------------------------------------------------
app = QtWidgets.QApplication([])

pg.setConfigOptions(background='k', foreground='w', useOpenGL=True, antialias=True)

win = pg.GraphicsLayoutWidget(show=True, title="DTLN Hearing Aid - Real-Time")
win.resize(1200, 900)

# Row 1: Input waveform
p_in = win.addPlot(title="Input Waveform")
p_in.setLabel('bottom', 'Time', units='s')
p_in.setLabel('left', 'Amplitude')
p_in.setYRange(-1.0, 1.0)
curve_in = p_in.plot(wave_time, wave_in_hist, pen=pg.mkPen('g'))

# Row 2: Output waveform
win.nextRow()
p_out = win.addPlot(title="Output Waveform")
p_out.setLabel('bottom', 'Time', units='s')
p_out.setLabel('left', 'Amplitude')
p_out.setYRange(-1.0, 1.0)
curve_out = p_out.plot(wave_time, wave_out_hist, pen=pg.mkPen('y'))

# Row 3: Input spectrogram
win.nextRow()
p_spec_in = win.addPlot(title="Input Spectrogram (dB)")
p_spec_in.setLabel('bottom', f"Frames (~{block_shift_ms:.0f} ms each)")
p_spec_in.setLabel('left', 'Freq', units='Hz')
img_in = pg.ImageItem()
img_in.setLookupTable(lut)
p_spec_in.addItem(img_in)
p_spec_in.setLimits(xMin=0, xMax=spec_cols, yMin=0, yMax=fs_target/2)
p_spec_in.setAspectLocked(False)

# Row 4: Output spectrogram
win.nextRow()
p_spec_out = win.addPlot(title="Output Spectrogram (dB)")
p_spec_out.setLabel('bottom', f"Frames (~{block_shift_ms:.0f} ms each)")
p_spec_out.setLabel('left', 'Freq', units='Hz')
img_out = pg.ImageItem()
img_out.setLookupTable(lut)
p_spec_out.addItem(img_out)
p_spec_out.setLimits(xMin=0, xMax=spec_cols, yMin=0, yMax=fs_target/2)
p_spec_out.setAspectLocked(False)

# Row 5: Spectral Difference Heatmap
win.nextRow()
p_spec_diff = win.addPlot(title="Difference Heatmap (Input − Output dB)")
p_spec_diff.setLabel('bottom', f"Frames (~{block_shift_ms:.0f} ms each)")
p_spec_diff.setLabel('left', 'Freq', units='Hz')
img_diff = pg.ImageItem()
# Use same LUT but you could choose another (e.g., 'plasma') by creating new
img_diff.setLookupTable(lut)
p_spec_diff.addItem(img_diff)
p_spec_diff.setLimits(xMin=0, xMax=spec_cols, yMin=0, yMax=fs_target/2)
p_spec_diff.setAspectLocked(False)

# Optional grid
if not cli.no_grid:
    for p in (p_spec_in, p_spec_out, p_spec_diff):
        p.showGrid(x=False, y=True, alpha=0.3)

# Freq ticks (kHz labels every 1k)
freq_ticks_major = [(k * 1000, f"{k}k" if k else "0") for k in range(0, (fs_target // 2000) * 2 + 1)]
# For 16k fs -> 0..8k -> k in 0..8
freq_ticks_minor = [(k * 500, "") for k in range(1, (fs_target // 1000) * 1)]  # half steps unlabeled
for p in (p_spec_in, p_spec_out, p_spec_diff):
    ax = p.getAxis('left')
    ax.setTicks([freq_ticks_major, freq_ticks_minor])

# ------------------------------------------------------------------
# Audio callback (real-time thread)
# ------------------------------------------------------------------
def audio_callback(indata, outdata, frames, time_info, status):
    global in_buffer, out_buffer, states_1, states_2
    global wave_in_hist, wave_out_hist, spec_in_hist, spec_out_hist

    if status:
        print(status, flush=True)

    if frames != block_shift:
        outdata.fill(0.0)
        return

    input_chunk = np.squeeze(indata).astype(np.float32)

    # Update rolling input frame
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = input_chunk

    # FFT
    in_fft   = np.fft.rfft(in_buffer)
    in_mag   = np.abs(in_fft).astype(np.float32)
    in_phase = np.angle(in_fft)

    # Core 1
    in_mag_reshaped = in_mag.reshape(1, 1, -1)
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag_reshaped)
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.invoke()
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

    # Apply mask & inverse FFT
    est_complex = in_mag * out_mask.squeeze() * np.exp(1j * in_phase)
    est_block = np.fft.irfft(est_complex).astype(np.float32)

    # Core 2
    est_block_reshaped = est_block.reshape(1, 1, -1)
    interpreter_2.set_tensor(input_details_2[0]['index'], est_block_reshaped)
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.invoke()
    out_block = interpreter_2.get_tensor(output_details_2[0]['index']).squeeze()
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])

    # Overlap-add
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = 0.0
    out_buffer += out_block

    # Output
    out_chunk = out_buffer[:block_shift]
    outdata[:] = out_chunk.reshape(-1, 1)

    # --- Update history for GUI ---
    with hist_lock:
        wave_in_hist  = np.roll(wave_in_hist,  -block_shift)
        wave_in_hist[-block_shift:] = input_chunk

        wave_out_hist = np.roll(wave_out_hist, -block_shift)
        wave_out_hist[-block_shift:] = out_chunk

        spec_in_hist  = np.roll(spec_in_hist,  -1, axis=1)
        spec_out_hist = np.roll(spec_out_hist, -1, axis=1)
        spec_in_hist[:,  -1] = in_mag
        spec_out_hist[:, -1] = np.abs(np.fft.rfft(out_buffer))


# ------------------------------------------------------------------
# GUI updater (Qt timer, main thread)
# ------------------------------------------------------------------
def gui_update():
    with hist_lock:
        w_in  = wave_in_hist.copy()
        w_out = wave_out_hist.copy()
        s_in  = spec_in_hist.copy()
        s_out = spec_out_hist.copy()

    # Waveforms
    curve_in.setData(wave_time, w_in)
    curve_out.setData(wave_time, w_out)

    # Spectrograms: mag->dB
    s_in_db  = mag2db(s_in)
    s_out_db = mag2db(s_out)

    # Auto contrast?
    if cli.auto_contrast:
        # Robust percentiles to avoid spikes
        in_lo, in_hi   = np.percentile(s_in_db,  5), np.percentile(s_in_db,  99)
        out_lo, out_hi = np.percentile(s_out_db, 5), np.percentile(s_out_db, 99)
        lev_in  = (in_lo,  in_hi)
        lev_out = (out_lo, out_hi)
    else:
        # Use fixed CLI range
        lev_in = lev_out = db_range

    # Clip for display
    s_in_db  = np.clip(s_in_db,  lev_in[0],  lev_in[1])
    s_out_db = np.clip(s_out_db, lev_out[0], lev_out[1])

    # Update spectrogram images
    img_in.setImage(s_in_db,  levels=lev_in,  autoLevels=False)
    img_out.setImage(s_out_db, levels=lev_out, autoLevels=False)

    # Map rows to Hz (0..fs/2) & cols to frames
    rect = QtCore.QRectF(0, 0, spec_cols, fs_target/2)
    img_in.setRect(rect)
    img_out.setRect(rect)

    # -------- Difference Heatmap --------
    # dB reduction: if input>output, positive reduction; else 0
    # reduction = in_db - out_db (both already in dB)
    # or more robust ratio: 20log10( (s_in+eps) / (s_out+eps) )
    eps = 1e-10
    diff_db = 20.0 * np.log10((s_in + eps) / (s_out + eps))
    diff_db = np.clip(diff_db, 0.0, diff_max)  # show only attenuation
    img_diff.setImage(diff_db, levels=(0.0, diff_max), autoLevels=False)
    img_diff.setRect(rect)


# ------------------------------------------------------------------
# Audio stream control
# ------------------------------------------------------------------
stream = None

def start_audio_stream():
    global stream
    stream = sd.Stream(
        device=(cli.input_device, cli.output_device),
        samplerate=fs_target,
        blocksize=block_shift,
        dtype=np.float32,
        latency=cli.latency,
        channels=1,
        callback=audio_callback,
    )
    stream.start()
    print("#" * 80)
    print("DTLN real-time running with enhanced PyQtGraph visualization.")
    print("Close the window or press Ctrl+C in terminal to stop.")
    print("#" * 80)

def stop_audio_stream():
    global stream
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        stream = None


# ------------------------------------------------------------------
# Graceful exit
# ------------------------------------------------------------------
def handle_sigint(sig, frame):
    print("\n[INFO] SIGINT received. Exiting ...")
    stop_audio_stream()
    QtWidgets.QApplication.quit()

signal.signal(signal.SIGINT, handle_sigint)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    start_audio_stream()

    # periodic GUI update
    timer = QtCore.QTimer()
    timer.timeout.connect(gui_update)
    timer.start(int(cli.plot_interval * 1000))

    # run Qt loop
    exit_code = app.exec_()

    stop_audio_stream()
    print("[INFO] Shutdown complete.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
