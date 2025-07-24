import os
import tensorflow as tf
from DTLN_model import DTLN_model, audio_generator
import soundfile as sf
import numpy as np

class TrainingConfig:
    def __init__(self):
        self.run_name = "dtln_model_pretrained"
        self.path_to_train_mix = "dataset_16k/processed/noisy_train"
        self.path_to_train_speech = "dataset_16k/processed/clean_train"
        self.path_to_val_mix = "dataset_16k/processed/noisy_val"
        self.path_to_val_speech = "dataset_16k/processed/clean_val"
        self.model_dir = os.path.join("./trained_models/", "dtln_model.h5")
        self.initial_epoch = 0
        self.max_epochs = 200
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.block_len = 512
        self.block_shift = 128
        self.len_samples = 5  # 5 seconds at 16 kHz

    def validate_paths(self):
        for path in [self.path_to_train_mix, self.path_to_train_speech, self.path_to_val_mix, self.path_to_val_speech]:
            if not os.path.exists(path):
                raise ValueError(f"Directory not found: {path}")
            wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
            if not wav_files:
                raise ValueError(f"No WAV files found in {path}")
            valid_files = []
            for file in wav_files:
                file_path = os.path.join(path, file)
                try:
                    audio, fs = sf.read(file_path)
                    if fs != 16000 or audio.ndim != 1 or len(audio) < self.len_samples * 16000:
                        continue
                    valid_files.append(file)
                except Exception:
                    continue
            if not valid_files:
                raise ValueError(f"No valid WAV files in {path}")

def train_model():
    config = TrainingConfig()

    try:
        config.validate_paths()
        os.makedirs(os.path.dirname(config.model_dir), exist_ok=True)
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        raise

    model = DTLN_model()
    model.batchsize = config.batch_size
    model.lr = config.learning_rate
    model.len_samples = config.len_samples
    model.blockLen = config.block_len
    model.block_shift = config.block_shift
    model.build_DTLN_model(norm_stft=False)
    print(f"Starting training for {config.run_name}")

    len_in_samples = int(np.fix(model.fs * model.len_samples / model.block_shift) * model.block_shift)
    generator_input = audio_generator(path_to_input=config.path_to_train_mix,
                                     path_to_s1=config.path_to_train_speech,
                                     len_of_samples=len_in_samples,
                                     fs=model.fs,
                                     batchsize=config.batch_size,
                                     train_flag=True)
    dataset = generator_input.tf_data_set
    steps_train = max(generator_input.total_samples // config.batch_size, 1)

    generator_val = audio_generator(path_to_input=config.path_to_val_mix,
                                   path_to_s1=config.path_to_val_speech,
                                   len_of_samples=len_in_samples,
                                   fs=model.fs,
                                   batchsize=config.batch_size)
    dataset_val = generator_val.tf_data_set
    steps_val = max(generator_val.total_samples // config.batch_size, 1)

    print(f"Starting fit: steps_train={steps_train}, steps_val={steps_val}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
    model.model.compile(loss=model.lossWrapper(), optimizer=optimizer)
    model.model.fit(
        x=dataset,
        batch_size=config.batch_size,
        steps_per_epoch=steps_train,
        epochs=config.max_epochs,
        verbose=1,
        validation_data=dataset_val,
        validation_steps=steps_val
    )

    # Save only the final model as .h5
    model.model.save(config.model_dir)
    print(f"Training completed. Final model saved to {config.model_dir}")

if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    train_model()