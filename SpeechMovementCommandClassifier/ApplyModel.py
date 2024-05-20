from SpeechMovementCommandClassifier import *

import tensorflow as tf
import wave
import pyaudio
import sounddevice
import numpy as np
import zmq

def main():

    label_names = ["up", "right", "down", "left", "unkown"]

    classifier = SpeechMovementCommandClassifier()

    x = tf.zeros(shape=(32, 124, 129, 1))
    classifier(x)
        
    classifier.build(input_shape=(1, 124, 129, 1))
    classifier.summary()

    classifier.load_weights(f"./saved_models/epoch_20.weights.h5")

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=3200
    )
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    while True:
        
        # Record wave
        wav = []
        for i in range(int(16000/3200)):
            data = stream.read(3200)
            wav.append(data)

        wav = np.frombuffer(b"".join(wav), dtype=np.int16)
        
        # Wave -> Spectrogram
        wav = tf.cast(wav, tf.float32)
        spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)

        spectrogram = tf.abs(spectrogram)

        # Add channel dim
        spectrogram = tf.expand_dims(spectrogram, axis=-1)

        # Add batch dim
        spectrogram = tf.expand_dims(spectrogram, axis=0)

        prediction = classifier(spectrogram)

        # Remove batch dim
        prediction = prediction[0]

        
        prediction = tf.argmax(prediction).numpy()
        word = label_names[prediction]
        print(word)
        socket.send_string(word)
        



    # stream.stop_stream()
    # stream.close()

    # p.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")