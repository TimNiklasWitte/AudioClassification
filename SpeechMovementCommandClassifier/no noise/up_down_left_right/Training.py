import tensorflow as tf
import tqdm
import datetime
import os 

from Classifier import *

NUM_EPOCHS = 20
BATCH_SIZE = 64


# alphanumeric order
labels = sorted(os.listdir("./../../speech_commands_v0.02"))
idx_label_dict = {label: idx for idx, label in enumerate(labels)}
    
up_label = idx_label_dict["up"]
down_label = idx_label_dict["down"]
right_label = idx_label_dict["right"]
left_label = idx_label_dict["left"]


def main():
 
    #
    # Load dataset
    #   
    
    train_ds, test_ds = tf.keras.utils.audio_dataset_from_directory(
                            directory="./../../speech_commands_v0.02",
                            batch_size=None,
                            validation_split=0.1,
                            seed=0,
                            output_sequence_length=16000,
                            subset='both'
                        )
    
    train_ds = train_ds.apply(prepare_data)
    test_ds = test_ds.apply(prepare_data)
    
 
    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Initialize model
    #

    classifier = Classifier()

    x = tf.zeros(shape=(32, 124, 129, 1))
    classifier(x)
    
    classifier.build(input_shape=(1, 124, 129, 1))
    classifier.summary()

    #
    # Train and test loss/accuracy
    #
    print(f"Epoch 0")
    log(train_summary_writer, classifier, train_ds, test_ds, 0)

    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for x, target in tqdm.tqdm(train_ds, position=0, leave=True): 
            classifier.train_step(x, target)

        log(train_summary_writer, classifier, train_ds, test_ds, epoch)

        if epoch % 10 == 0:
            # Save model (its parameters)
            classifier.save_weights(f"./saved_models/epoch_{epoch}.weights.h5")


def log(train_summary_writer, classifier, train_dataset, test_dataset, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        classifier.test_step(train_dataset.take(500))

    label_names = ["up", "right", "down", "left", "unkown"]
    with train_summary_writer.as_default():

        #
        # Train
        #
        for metric in classifier.metrics:

            if isinstance(metric, tf.keras.metrics.F1Score):
                f1_scores = metric.result()
                for idx, label in enumerate(label_names):
                    print(f"train_{metric.name}_{label}: {f1_scores[idx]}")
                    tf.summary.scalar(f"train_{metric.name}_{label}", f1_scores[idx], step=epoch)

            else:
                tf.summary.scalar(f"train_{metric.name}", metric.result(), step=epoch)
                print(f"train_{metric.name}: {metric.result()}")

            metric.reset_state()


        #
        # Test
        #

        classifier.test_step(test_dataset)

        for metric in classifier.metrics:
            if isinstance(metric, tf.keras.metrics.F1Score):
                f1_scores = metric.result()
                for idx, label in enumerate(label_names):
                    print(f"test_{metric.name}_{label}: {f1_scores[idx]}")
                    tf.summary.scalar(f"test_{metric.name}_{label}", f1_scores[idx], step=epoch)

            else:
                tf.summary.scalar(f"test_{metric.name}", metric.result(), step=epoch)
                print(f"test_{metric.name}: {metric.result()}")




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

def prepare_data(dataset):

    dataset = dataset.map(lambda wav, label: (wav, relabel(label)))

    # Flatten: (num_samples, 1) -> (num_samples)
    dataset = dataset.map(lambda wav, label: (tf.reshape(wav, (-1,)), label))

    # Spectrogram (height, width, 1)
    dataset = dataset.map(lambda wav, label:(get_spectrogram(wav), label))

    # One hot target
    dataset = dataset.map(lambda spectrogram, label: (spectrogram, tf.one_hot(label, depth=5)))

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")