import tensorflow as tf
import tqdm
import datetime

from Classifier import *

NUM_EPOCHS = 100
BATCH_SIZE = 64

def main():
 
    #
    # Load dataset
    #   
    
    train_ds, test_ds = tf.keras.utils.audio_dataset_from_directory(
                            directory="./mini_speech_commands",
                            batch_size=None,
                            validation_split=0.2,
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

        if epoch % 5 == 0:
            # Save model (its parameters)
            classifier.save_weights(f"./saved_models/epoch_{epoch}.weights.h5")


def log(train_summary_writer, classifier, train_dataset, test_dataset, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        classifier.test_step(train_dataset.take(5000))

    #
    # Train
    #
    train_loss = classifier.metric_loss.result()
    train_accuracy = classifier.metric_accuracy.result()

    classifier.metric_loss.reset_state()
    classifier.metric_accuracy.reset_state()

    #
    # Test
    #

    classifier.test_step(test_dataset)

    test_loss = classifier.metric_loss.result()
    test_accuracy = classifier.metric_accuracy.result()

    classifier.metric_loss.reset_state()
    classifier.metric_accuracy.reset_state()

    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss", train_loss, step=epoch)
        tf.summary.scalar(f"train_accuracy", train_accuracy, step=epoch)

        tf.summary.scalar(f"test_loss", test_loss, step=epoch)
        tf.summary.scalar(f"test_accuracy", test_accuracy, step=epoch)

    #
    # Output
    #
    print(f"    train_loss: {train_loss}")
    print(f"     test_loss: {test_loss}")
    print(f"train_accuracy: {train_accuracy}")
    print(f" test_accuracy: {test_accuracy}")



def get_spectrogram(wav):

    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    
    return spectrogram

 
def prepare_data(dataset):

    # Flatten: (num_samples, 1) -> (num_samples)
    dataset = dataset.map(lambda wav, label: (tf.reshape(wav, (-1,)), label))

    # Spectrogram (height, width, 1)
    dataset = dataset.map(lambda wav, label:(get_spectrogram(wav), label))

    # One hot target
    dataset = dataset.map(lambda spectrogram, label: (spectrogram, tf.one_hot(label, depth=8)))

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")