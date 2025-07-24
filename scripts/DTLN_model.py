import os
import fnmatch
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import tensorflow as tf
import soundfile as sf
from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np
import pyaudio
import time

class ProcessStates(Layer):
    def __init__(self, num_layer, num_units, **kwargs):
        super(ProcessStates, self).__init__(**kwargs)
        self.num_layer = num_layer
        self.num_units = num_units

    def call(self, inputs):
        h_states, c_states = inputs
        h_stacked = tf.stack(h_states, axis=0)
        c_stacked = tf.stack(c_states, axis=0)
        h_reshaped = tf.reshape(h_stacked, [1, self.num_layer, self.num_units])
        c_reshaped = tf.reshape(c_stacked, [1, self.num_layer, self.num_units])
        return tf.stack([h_reshaped, c_reshaped], axis=-1)

class audio_generator:
    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, batchsize, train_flag=False):
        if not os.path.exists(path_to_input):
            raise FileNotFoundError(f"Input directory not found: {path_to_input}")
        if not os.path.exists(path_to_s1):
            raise FileNotFoundError(f"Target directory not found: {path_to_s1}")
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.batchsize = batchsize
        self.train_flag = train_flag
        self.count_samples()
        self.create_tf_data_obj()

    def count_samples(self):
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), "*.wav")
        if not self.file_names:
            raise ValueError(f"No WAV files found in {self.path_to_input}")
        clean_files = fnmatch.filter(os.listdir(self.path_to_s1), "*.wav")
        if set(self.file_names) != set(clean_files):
            raise ValueError("Mismatched filenames between noisy and clean directories")
        self.total_samples = 0
        skipped_files = []
        for file in self.file_names:
            file_path = os.path.join(self.path_to_input, file)
            try:
                audio, fs_read = sf.read(file_path)
                if fs_read != self.fs:
                    skipped_files.append((file, f"Sample rate {fs_read}, expected {self.fs}"))
                    continue
                if audio.ndim != 1:
                    skipped_files.append((file, f"Has {audio.ndim} channels, expected 1"))
                    continue
                if len(audio) < self.len_of_samples:
                    skipped_files.append((file, f"Too short: {len(audio)} samples, expected {self.len_of_samples}"))
                    continue
                self.total_samples += int(np.fix(len(audio) / self.len_of_samples))
            except Exception as e:
                skipped_files.append((file, f"Invalid file: {str(e)}"))
        if skipped_files:
            print(f"Warning: Skipped {len(skipped_files)} files in {self.path_to_input}:")
            for file, reason in skipped_files:
                print(f"  - {file}: {reason}")
        if self.total_samples == 0:
            raise ValueError(f"No valid samples found in {self.path_to_input}")

    def create_generator(self):
        if self.train_flag:
            shuffle(self.file_names)
        batch_noisy = []
        batch_speech = []
        for file in self.file_names:
            try:
                noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
                speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
                if fs_1 != self.fs or fs_2 != self.fs:
                    raise ValueError(f"Sampling rates do not match: noisy={fs_1}, speech={fs_2}, expected {self.fs}")
                if noisy.ndim != 1 or speech.ndim != 1:
                    raise ValueError(f"File {file} has multiple channels")
                if len(noisy) != len(speech):
                    raise ValueError(f"File {file} length mismatch: noisy={len(noisy)}, speech={len(speech)}")
                num_samples = int(np.fix(len(noisy) / self.len_of_samples))
                for idx in range(num_samples):
                    in_dat = noisy[int(idx * self.len_of_samples): int((idx + 1) * self.len_of_samples)]
                    tar_dat = speech[int(idx * self.len_of_samples): int((idx + 1) * self.len_of_samples)]
                    if np.any(np.isnan(in_dat)) or np.any(np.isinf(in_dat)) or np.any(np.isnan(tar_dat)) or np.any(np.isinf(tar_dat)):
                        print(f"Warning: NaN or Inf detected in file {file} at index {idx}, skipping batch")
                        continue
                    batch_noisy.append(in_dat)
                    batch_speech.append(tar_dat)
                    if len(batch_noisy) == self.batchsize:
                        yield np.array(batch_noisy).astype("float32"), np.array(batch_speech).astype("float32")
                        batch_noisy = []
                        batch_speech = []
            except Exception as e:
                print(f"Warning: Skipping file {file} in generator: {str(e)}")
                continue
        if batch_noisy:
            padded_batch_noisy = np.pad(batch_noisy, ((0, self.batchsize - len(batch_noisy)), (0, 0)), mode='constant')
            padded_batch_speech = np.pad(batch_speech, ((0, self.batchsize - len(batch_speech)), (0, 0)), mode='constant')
            yield padded_batch_noisy.astype("float32"), padded_batch_speech.astype("float32")

    def create_tf_data_obj(self):
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.batchsize, self.len_of_samples]),
                           tf.TensorShape([self.batchsize, self.len_of_samples])),
            args=None,
        ).repeat().prefetch(tf.data.AUTOTUNE)

class LogNormalization(Layer):
    def __init__(self, epsilon=1e-7, **kwargs):
        super(LogNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        return tf.math.log(inputs + self.epsilon)

class InstantLayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma = self.add_weight(shape=shape, initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(shape=shape, initializer="zeros", trainable=True, name="beta")

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma
        outputs = outputs + self.beta
        return outputs

class DTLN_model:
    def __init__(self):
        self.cost_function = self.snr_cost
        self.model = None
        self.fs = 16000
        self.batchsize = 16
        self.len_samples = 5
        self.activation = "sigmoid"
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 100
        self.encoder_size = 257
        self.eps = 1e-7
        os.environ["PYTHONHASHSEED"] = str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
            print("GPU acceleration enabled.")
        else:
            print("No GPU detected, using CPU for training.")

    @staticmethod
    def snr_cost(s_estimate, s_true):
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / (
                tf.reduce_mean(tf.math.square(s_true - s_estimate), axis=-1, keepdims=True) + 1e-6
        )
        snr = tf.clip_by_value(snr, 1e-6, 1e6)
        num = tf.math.log(snr)
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10 * (num / denom)
        return loss

    def lossWrapper(self):
        def lossFunction(y_true, y_pred):
            loss = tf.squeeze(self.cost_function(y_pred, y_true))
            loss = tf.reduce_mean(loss)
            return loss
        return lossFunction

    def stftLayer(self, x):
        x = tf.cast(x, tf.float32)
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        stft_dat = tf.signal.rfft(frames)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return [mag, phase]

    def fftLayer(self, x):
        x = tf.cast(x, tf.float32)
        frame = tf.expand_dims(x, axis=1)
        stft_dat = tf.signal.rfft(frame)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return [mag, phase]

    def ifftLayer(self, x):
        mag, phase = x
        s1_stft = tf.cast(mag, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))
        output = tf.signal.irfft(s1_stft)
        return tf.cast(output, tf.float32)

    def overlapAddLayer(self, x):
        return tf.signal.overlap_and_add(x, self.block_shift)

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        return mask

    def seperation_kernel_with_states(self, num_layer, mask_size, x, in_states):
        states_h = []
        states_c = []
        for idx in range(num_layer):
            in_state = [in_states[:, idx, :, 0], in_states[:, idx, :, 1]]
            x, h_state, c_state = LSTM(self.numUnits, return_sequences=True, unroll=True, return_state=True)(
                x, initial_state=in_state)
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
            states_h.append(h_state)
            states_c.append(c_state)
        out_states = ProcessStates(num_layer, self.numUnits)([states_h, states_c])
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        return mask, out_states

    def build_DTLN_model(self, norm_stft=False):
        time_dat = Input(shape=(None,), batch_size=self.batchsize)
        def stft_output_shape(input_shape):
            batch_size = self.batchsize
            time_steps = input_shape[1]
            num_frames = None
            freq_bins = self.blockLen // 2 + 1
            return [(batch_size, num_frames, freq_bins), (batch_size, num_frames, freq_bins)]
        mag, angle = Lambda(self.stftLayer, output_shape=stft_output_shape)(time_dat)
        if norm_stft:
            log_mag = LogNormalization()(mag)
            mag_norm = InstantLayerNormalization()(log_mag)
        else:
            mag_norm = mag
        mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen // 2 + 1), mag_norm)
        estimated_mag = Multiply()([mag, mask_1])
        def ifft_output_shape(input_shapes):
            batch_size = self.batchsize
            num_frames = input_shapes[0][1]
            return (batch_size, num_frames, self.blockLen)
        estimated_frames_1 = Lambda(self.ifftLayer, output_shape=ifft_output_shape)([estimated_mag, angle])
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
        estimated = Multiply()([encoded_frames, mask_2])
        decoded_frames = Conv1D(self.blockLen, 1, padding="causal", use_bias=False)(estimated)
        def overlap_add_output_shape(input_shape):
            batch_size = self.batchsize
            return (batch_size, None)
        estimated_sig = Lambda(self.overlapAddLayer, output_shape=overlap_add_output_shape)(decoded_frames)
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        self.compile_model()
        print(self.model.summary())

    def build_DTLN_model_stateful(self, norm_stft=False):
        time_dat = Input(batch_shape=(1, self.blockLen))
        states_input = Input(batch_shape=(1, self.numLayer, self.numUnits, 2))
        mag, angle = Lambda(self.fftLayer)(time_dat)
        if norm_stft:
            log_mag = LogNormalization()(mag)
            mag_norm = InstantLayerNormalization()(log_mag)
        else:
            mag_norm = mag
        mask_1, states_out_1 = self.seperation_kernel_with_states(self.numLayer, (self.blockLen // 2 + 1), mag_norm,
                                                                  states_input)
        estimated_mag = Multiply()([mag, mask_1])
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag, angle])
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        mask_2, states_out_2 = self.seperation_kernel_with_states(self.numLayer, self.encoder_size, encoded_frames_norm,
                                                                  states_out_1)
        estimated = Multiply()([encoded_frames, mask_2])
        decoded_frame = Conv1D(self.blockLen, 1, padding="causal", use_bias=False)(estimated)
        self.model = Model(inputs=[time_dat, states_input], outputs=[decoded_frame, states_out_2])
        self.compile_model()
        print(self.model.summary())

    def compile_model(self):
        optimizerAdam = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizerAdam)

    def create_saved_model(self, weights_file, target_name):
        if weights_file.find("_norm_") != -1:
            norm_stft = True
        else:
            norm_stft = False
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        self.model.load_weights(weights_file)
        tf.saved_model.save(self.model, target_name)

    def create_tf_lite_model(self, weights_file, target_name, use_dynamic_range_quant=False):
        # Adjust to match 4-layer architecture (1 LSTM layer per kernel)
        if weights_file.find("_norm_") != -1:
            norm_stft = True
            num_elements_first_core = 2 + 1 * 3 + 2  # 1 LSTM layer: LogNorm, InstNorm, LSTM, Dense, Activation
        else:
            norm_stft = False
            num_elements_first_core = 1 * 3 + 2  # 1 LSTM layer: Input, LSTM, Dense, Activation
        # Build model with 1 LSTM layer per kernel
        time_dat = Input(shape=(None,), batch_size=1)  # Batch size 1 for TFLite
        mag, angle = Lambda(self.stftLayer, output_shape=lambda x: [(1, None, self.blockLen // 2 + 1), (1, None, self.blockLen // 2 + 1)])(time_dat)
        if norm_stft:
            log_mag = LogNormalization()(mag)
            mag_norm = InstantLayerNormalization()(log_mag)
        else:
            mag_norm = mag
        # Use 1 LSTM layer for mask_1
        x = LSTM(self.numUnits, return_sequences=True)(mag_norm)
        mask_1 = Dense(self.blockLen // 2 + 1)(x)
        mask_1 = Activation(self.activation)(mask_1)
        estimated_mag = Multiply()([mag, mask_1])
        estimated_frames_1 = Lambda(self.ifftLayer, output_shape=lambda x: (1, None, self.blockLen))([estimated_mag, angle])
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # Use 1 LSTM layer for mask_2
        x = LSTM(self.numUnits, return_sequences=True)(encoded_frames_norm)
        mask_2 = Dense(self.encoder_size)(x)
        mask_2 = Activation(self.activation)(mask_2)
        estimated = Multiply()([encoded_frames, mask_2])
        decoded_frames = Conv1D(self.blockLen, 1, padding="causal", use_bias=False)(estimated)
        estimated_sig = Lambda(self.overlapAddLayer, output_shape=lambda x: (1, None))(decoded_frames)
        model = Model(inputs=time_dat, outputs=estimated_sig)
        model.load_weights(weights_file)  # Load weights into the adjusted model
        weights = model.get_weights()

        # First TFLite model (mask_1 with states)
        mag_input = Input(batch_shape=(1, 1, self.blockLen // 2 + 1))
        states_in_1 = Input(batch_shape=(1, 1, self.numUnits, 2))  # 1 layer state
        if norm_stft:
            log_mag = LogNormalization()(mag_input)
            mag_norm = InstantLayerNormalization()(log_mag)
        else:
            mag_norm = mag_input
        x = LSTM(self.numUnits, return_sequences=True, return_state=True)(mag_norm)
        mask_1, h_state_1, c_state_1 = x[0], x[1], x[2]  # Extract states
        mask_1 = Dense(self.blockLen // 2 + 1)(mask_1)
        mask_1 = Activation(self.activation)(mask_1)
        states_out_1 = ProcessStates(1, self.numUnits)([[h_state_1], [c_state_1]])
        model_1 = Model(inputs=[mag_input, states_in_1], outputs=[mask_1, states_out_1])

        # Second TFLite model (mask_2 with states)
        estimated_frame_input = Input(batch_shape=(1, 1, self.blockLen))
        states_in_2 = Input(batch_shape=(1, 1, self.numUnits, 2))  # 1 layer state
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frame_input)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        x = LSTM(self.numUnits, return_sequences=True, return_state=True)(encoded_frames_norm)
        mask_2, h_state_2, c_state_2 = x[0], x[1], x[2]  # Extract states
        mask_2 = Dense(self.encoder_size)(mask_2)
        mask_2 = Activation(self.activation)(mask_2)
        estimated = Multiply()([encoded_frames, mask_2])
        decoded_frame = Conv1D(self.blockLen, 1, padding="causal", use_bias=False)(estimated)
        states_out_2 = ProcessStates(1, self.numUnits)([[h_state_2], [c_state_2]])
        model_2 = Model(inputs=[estimated_frame_input, states_in_2], outputs=[decoded_frame, states_out_2])

        # Assign weights (approximate split based on 1 LSTM layer per core)
        model_1.set_weights(weights[:num_elements_first_core])
        model_2.set_weights(weights[num_elements_first_core:])

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_1 = converter.convert()
        with tf.io.gfile.GFile(target_name + "_1.tflite", "wb") as f:
            f.write(tflite_model_1)

        converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_2 = converter.convert()
        with tf.io.gfile.GFile(target_name + "_2.tflite", "wb") as f:
            f.write(tflite_model_2)

        print("TF Lite conversion complete!")

    def train_model(self, runName, path_to_train_mix, path_to_train_speech, path_to_val_mix, path_to_val_speech,
                    callbacks=None, initial_epoch=0):
        savePath = os.path.join("./models_", runName, "")
        os.makedirs(savePath, exist_ok=True)

        if callbacks is None:
            callbacks = []
        default_callbacks = [
            CSVLogger(os.path.join(savePath, f"training_{runName}.log")),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, cooldown=1),
            EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(savePath, f"{runName}_epoch_{{epoch:02d}}.weights.h5"),
                monitor="val_loss",
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode="auto",
                save_freq="epoch"
            )
        ]
        callbacks = default_callbacks + callbacks if callbacks else default_callbacks

        try:
            self.build_DTLN_model(norm_stft=False)
            print(f"Starting training for {runName} with optimized architecture")

            len_in_samples = int(np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift)
            generator_input = audio_generator(path_to_train_mix, path_to_train_speech, len_in_samples, self.fs,
                                              self.batchsize, train_flag=True)
            dataset = generator_input.tf_data_set
            steps_train = max(generator_input.total_samples // self.batchsize, 1)

            generator_val = audio_generator(path_to_val_mix, path_to_val_speech, len_in_samples, self.fs,
                                            self.batchsize)
            dataset_val = generator_val.tf_data_set
            steps_val = max(generator_val.total_samples // self.batchsize, 1)

            print(f"Starting fit: steps_train={steps_train}, steps_val={steps_val}")
            self.model.fit(
                x=dataset,
                batch_size=self.batchsize,
                steps_per_epoch=steps_train,
                epochs=self.max_epochs,
                verbose=1,
                validation_data=dataset_val,
                validation_steps=steps_val,
                callbacks=callbacks,
                initial_epoch=initial_epoch
            )
            tf.keras.backend.clear_session()

            latest_model = os.path.join(savePath, f"{runName}_epoch_{self.max_epochs - 1:02d}.weights.h5")
            old_models = [f for f in os.listdir(savePath) if
                          f.endswith('.weights.h5') and f != os.path.basename(latest_model)]
            for old_model in old_models:
                os.remove(os.path.join(savePath, old_model))
            print(f"Replaced old models with {latest_model}")
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            interrupt_checkpoint = os.path.join(savePath, f"interrupt_checkpoint_{initial_epoch}.weights.h5")
            self.model.save_weights(interrupt_checkpoint)
            raise

    def process_real_time(self, weights_file, output_device=None):
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Weights file not found: {weights_file}")
        norm_stft = weights_file.find("_norm_") != -1
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        self.model.load_weights(weights_file)

        p = pyaudio.PyAudio()
        sample_rate = self.fs
        chunk_size = self.blockLen

        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        output=True if output_device is not None else False)

        states = np.zeros((1, self.numLayer, self.numUnits, 2))

        print("Starting real-time audio processing. Press Ctrl+C to stop.")
        try:
            while True:
                audio_data = stream.read(chunk_size, exception_on_overflow=False)
                audio_float = np.frombuffer(audio_data, dtype=np.float32)
                if len(audio_float) < chunk_size:
                    audio_float = np.pad(audio_float, (0, chunk_size - len(audio_float)), mode='constant')
                audio_input = audio_float.reshape(1, chunk_size)
                predicted_frame, new_states = self.model.predict([audio_input, states], verbose=0)
                states = new_states
                output_audio = predicted_frame.flatten()
                if output_device is not None:
                    stream.write(output_audio.tobytes())
                time.sleep(chunk_size / sample_rate)
        except KeyboardInterrupt:
            print("Stopping real-time processing.")
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Real-time processing error: {str(e)}")
            stream.stop_stream()
            stream.close()
            p.terminate()
            raise

if __name__ == "__main__":
    model = DTLN_model()
    model.build_DTLN_model()