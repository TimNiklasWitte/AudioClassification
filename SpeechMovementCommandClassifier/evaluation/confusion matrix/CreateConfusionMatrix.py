
import sys
sys.path.append("../..")

from SpeechMovementCommandClassifier import *


import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import librosa
import os

BATCH_SIZE = 64

# alphanumeric order
labels = sorted(os.listdir("./../../speech_commands_v0.02"))

idx_label_dict = {label: idx for idx, label in enumerate(labels)}
    
up_label = idx_label_dict["up"]
down_label = idx_label_dict["down"]
right_label = idx_label_dict["right"]
left_label = idx_label_dict["left"]

idel_wav_file_path = "./../../../LESSI noise/Idel.wav"
idel_wav, sr = librosa.load(idel_wav_file_path)
idel_wav = librosa.resample(y=idel_wav, orig_sr=sr, target_sr=16000)

walk_wav_file_path = "./../../../LESSI noise/Walk.wav"
walk_wav, sr = librosa.load(walk_wav_file_path)
walk_wav = walk_wav[50000:]
walk_wav = librosa.resample(y=walk_wav, orig_sr=sr, target_sr=16000)

def get_spectrogram(wav):

    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    
    return spectrogram

def relabel(label):

    # up: 0
    # right: 1
    # down: 2
    # left: 3
    # unknown: 4

    if label == up_label:
        return 0
    elif label == right_label:
        return 1
    elif label == down_label:
        return 2
    elif label == left_label:
        return 3
    else:
        return 4


def create_pipeline(alpha, noise_type):

    if noise_type == "idel":
        r = 0
    elif noise_type == "walk":
        r = 1
    else:
        exit(0)


    def add_noise(speech_wav):

        # idel
        if r == 0:
        
            pos = tf.random.uniform(shape=(), minval=0, maxval=len(idel_wav) - 16000, dtype=tf.int32)
            noise_indices = tf.range(pos, pos+16000, dtype=tf.dtypes.int64)
            noise = tf.gather(idel_wav, noise_indices)

            superimposed = alpha * speech_wav + (1 - alpha) * noise
            return superimposed
        
        # walk
        else:
            
            pos = tf.random.uniform(shape=(), minval=0, maxval=len(walk_wav) - 16000, dtype=tf.int32)
            noise_indices = tf.range(pos, pos+16000, dtype=tf.dtypes.int64)
            noise = tf.gather(walk_wav, noise_indices)

            superimposed = alpha * speech_wav + (1 - alpha) * noise
            return superimposed




    def prepare_data(dataset):

        dataset = dataset.map(lambda wav, label: (wav, relabel(label)))

        # Flatten: (num_samples, 1) -> (num_samples)
        dataset = dataset.map(lambda wav, label: (tf.reshape(wav, (-1,)), label))
        
        # One hot target
        dataset = dataset.map(lambda wav, label: (wav, tf.one_hot(label, depth=5)))


        #
        # Cache
        #

        dataset = dataset.cache()

        dataset = dataset.map(lambda wav, target: (add_noise(wav), target))

        # Spectrogram shape = (124, 129, 1)
        dataset = dataset.map(lambda wav, target:(get_spectrogram(wav), target))

        # resize: (124, 129) -> (128, 128)
        dataset = dataset.map(lambda spectrogram, target: (tf.image.resize(spectrogram, (128, 128)), target))

        #
        # Shuffle, batch, prefetch
        #
        dataset = dataset.shuffle(5000)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    return prepare_data






def main():

    label_names = ["forward", "right", "backward", "left", "unkown"]

    classifier = SpeechMovementCommandClassifier()

    x = tf.zeros(shape=(32, 124, 129, 1))
    classifier(x)
        
    classifier.build(input_shape=(1, 124, 129, 1))
    classifier.summary()

    classifier.load_weights(f"./../../saved_models/epoch_30.weights.h5")

    train_ds, test_ds = tf.keras.utils.audio_dataset_from_directory(
                            directory="./../../speech_commands_v0.02",
                            batch_size=None,
                            validation_split=0.2,
                            seed=0,
                            output_sequence_length=16000,
                            subset='both'
                        )
    
    for noise_type in ["idel", "walk"]:
        for alpha in np.arange(0.1, 0.00, -0.01): #np.arange(1, 0.0, -0.1): # np.arange(0.1, 0.00, -0.01)

            preprocessing_pipeline = create_pipeline(alpha=alpha, noise_type=noise_type)

            test_ds_preprocessed = test_ds.apply(preprocessing_pipeline)

            y_pred = []
            y_true = []

            print(noise_type, f"{alpha:.2f}")
            for x, target in tqdm.tqdm(test_ds_preprocessed, position=0, leave=True):

                y_true.append(np.argmax(target, axis=-1))
        
                preds = classifier(x)

                y_pred.append(np.argmax(preds, axis=- 1))

            correct_labels = tf.concat([item for item in y_true], axis=0)
            predicted_labels = tf.concat([item for item in y_pred], axis=0)
            
        

            confusion_matrix = tf.math.confusion_matrix(predicted_labels, correct_labels)
            with open(f"./absolute/{noise_type}/ConfusionMatrix_{alpha:.2f}", "w") as file:
                # Writing data to a file
                file.write(str(confusion_matrix))
            
            
            # Normalize
            num_labels_per_class = tf.reduce_sum(confusion_matrix, axis=1)
            num_labels_per_class = tf.expand_dims(num_labels_per_class, axis=1)
            confusion_matrix = confusion_matrix / num_labels_per_class


            confusion_matrix_np = confusion_matrix.numpy()
            np.save(f"./relative numpy/{noise_type}/ConfusionMatrix_{alpha:.2f}.npy", confusion_matrix_np)

            df_cm = pd.DataFrame(confusion_matrix, index = label_names, columns = label_names)
            heatmap = sn.heatmap(df_cm, annot=True)
            heatmap.set_xlabel("Predicted label")
            heatmap.set_ylabel("True label")

            plt.title(f"{noise_type}, alpha = {alpha:.2f}")
            plt.savefig(f"./plots/{noise_type}/ConfusionMatrix_{alpha:.2f}.png")

            plt.clf()
            plt.close()


    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")