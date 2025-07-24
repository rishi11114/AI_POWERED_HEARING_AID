# AI-Powered Hearing Aid with DTLN
## Project Overview
This project is an AI-powered hearing aid that uses the Dual-signal Transformation LSTM Network (DTLN) for real-time noise suppression. The DTLN model intelligently filters out background noise, enhancing speech clarity and providing a superior listening experience in diverse environments. The model is lightweight and efficient, making it ideal for deployment on embedded systems.

## Key Features
Real-time Noise Suppression: Uses the DTLN model for instant noise reduction.

Speech Enhancement: Focuses on clarifying human speech for better intelligibility.

High Performance: Achieves state-of-the-art results with competitive performance metrics.

Efficient Model: The DTLN network has less than one million parameters, ensuring low latency and real-time capability on devices like the Raspberry Pi.

Python-based: The project is built using Python with the TensorFlow 2.x framework.

## Performance Highlights
Execution Time: The model can run in as little as 2.2 ms on a Raspberry Pi 3 B+.

Speech Quality (MOS): Outperforms the DNS-Challenge baseline by 0.24 points.

Audio Enhancement: Effectively enhances audio in noisy environments, including music, traffic, and wind.

## Getting Started
To get the DTLN model running, you will need to install the required Python dependencies and use the evaluation script.

## Prerequisites
TensorFlow (2.x)

librosa

wavinfo

### Running an Evaluation
Use the following command to process a folder of audio files with the pretrained model:

Bash

$ python run_evaluation.py -i /path/to/input -o /path/for/processed -m ./pretrained_model/model.h5

