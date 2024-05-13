from Classifier import *
from Training import *

import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def main():

    label_names = ["forward", "right", "backward", "left", "unkown"]

    classifier = Classifier()

    x = tf.zeros(shape=(32, 124, 129, 1))
    classifier(x)
        
    classifier.build(input_shape=(1, 124, 129, 1))
    classifier.summary()

    classifier.load_weights(f"./saved_models/epoch_20.weights.h5")

    train_ds, test_ds = tf.keras.utils.audio_dataset_from_directory(
                            directory="./speech_commands_v0.02",
                            batch_size=None,
                            validation_split=0.2,
                            seed=0,
                            output_sequence_length=16000,
                            subset='both'
                        )

    train_ds = train_ds.apply(prepare_data)
    test_ds = test_ds.apply(prepare_data)

    y_pred = []
    y_true = []

    for x, target in tqdm.tqdm(test_ds, position=0, leave=True):

        y_true.append(np.argmax(target, axis=-1))
       
        preds = classifier(x)

        y_pred.append(np.argmax(preds, axis=- 1))

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    
  

    confusion_matrix = tf.math.confusion_matrix(predicted_labels, correct_labels)

    

    print(confusion_matrix)

    # Normalize
    num_labels_per_class = tf.reduce_sum(confusion_matrix, axis=1)
    num_labels_per_class = tf.expand_dims(num_labels_per_class, axis=1)
    confusion_matrix = confusion_matrix / num_labels_per_class
  
    df_cm = pd.DataFrame(confusion_matrix, index = label_names, columns = label_names)
    heatmap = sn.heatmap(df_cm, annot=True)
    heatmap.set_xlabel("Predicted label")
    heatmap.set_ylabel("True label")

    plt.savefig("ConfusionMatrix.png")
    plt.show()
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")