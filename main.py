import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def spectral_subtraction(input_wav, output_wav, sr=44100):
    # Read the WAV file
    fs, data = wavfile.read(input_wav)
    plt.subplot(221)
    plt.plot(data)
    plt.title("Input signal with noise")

    # Normalize data if it's stereo
    if len(data.shape) == 2:
        data = data.mean(axis=1)  # Convert to mono if stereo

    # Set parameters
    frame_size = 2048
    hop_size = 512

    # Apply a Hanning window
    window = np.hanning(frame_size)

    # Compute the STFT (Short-Time Fourier Transform)
    f, t, Zxx = signal.stft(data, fs=fs, window=window, nperseg=frame_size, noverlap=hop_size)
    plt.subplot(223)
    plt.plot(np.abs(Zxx))
    plt.title("Magnitude spectrum of input")
    """
    plt.plot(np.angle(Zxx))
    plt.title("Phase spectrum")
    plt.show()
    """

    # Estimate the noise (assuming first few frames are noise)
    noise_estimation = np.mean(np.abs(Zxx[:, :10]), axis=1, keepdims=True)  # Average first 10 frames

    # Spectral Subtraction
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    magnitude = np.maximum(magnitude - noise_estimation, 0)  # Subtract noise estimation
    
    
    
    # Reconstruct the signal using inverse STFT
    plt.subplot(224)
    Zxx_clean = magnitude * np.exp(1j * phase)
    plt.plot(np.abs(Zxx_clean))
    plt.title("Magnitude spectrum of output")
    _, data_clean = signal.istft(Zxx_clean, fs=fs, window=window, nperseg=frame_size, noverlap=hop_size)

    # Write the filtered data to a new WAV file
    wavfile.write(output_wav, fs, data_clean.astype(np.int16))
    plt.subplot(222)
    plt.plot(data_clean)
    plt.title("Output Signal without noise")
    plt.tight_layout()
    plt.show()

# Example usage
input_file = 'input.wav'  # Path to your input WAV file
output_file = 'output_final.wav'  # Path to save the output WAV file
spectral_subtraction(input_file, output_file)

input_file = 'input1.wav'  # Path to your input WAV file
output_file = 'output_final1.wav'  # Path to save the output WAV file
spectral_subtraction(input_file, output_file)

input_file = 'input2.wav'  # Path to your input WAV file
output_file = 'output_final2.wav'  # Path to save the output WAV file
spectral_subtraction(input_file, output_file)

input_file = 'input3.wav'  # Path to your input WAV file
output_file = 'output_final3.wav'  # Path to save the output WAV file
spectral_subtraction(input_file, output_file)
