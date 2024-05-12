import tensorflow as tf
import numpy as np
import os 
import matplotlib.pyplot as plt

# alphanumeric order
labels = sorted(os.listdir("./speech_commands_v0.02"))

idx_label_dict = {label: idx for idx, label in enumerate(labels)}
    
forward_label = idx_label_dict["forward"]
backward_label = idx_label_dict["backward"]
right_label = idx_label_dict["right"]
left_label = idx_label_dict["left"]

labels = ["forward", "right", "backward", "left", "unknown"]

def main():
    train_ds, test_ds = tf.keras.utils.audio_dataset_from_directory(
                            directory="./speech_commands_v0.02",
                            batch_size=None,
                            validation_split=0.1,
                            seed=0,
                            output_sequence_length=16000,
                            subset='both'
                        )
    
    train_ds = train_ds.apply(prepare_data)
    test_ds = test_ds.apply(prepare_data)
    
    x = np.arange(0, 5)

    #
    # Train
    #

    label_cnt = np.zeros(shape=(5,), dtype=np.uint64)
    for wav, label in train_ds:
        label_cnt[label] += 1
    
    fig, axs = plt.subplots(nrows=2, ncols=1)

    axs[0].set_title("Train dataset")
    axs[0].bar(x, height=label_cnt, label=labels)

    #
    # Test
    #

    label_cnt = np.zeros(shape=(5,), dtype=np.uint64)
    for wav, label in test_ds:
        label_cnt[label] += 1
    
    axs[1].set_title("Test dataset")
    axs[1].bar(x, height=label_cnt, label=labels)

    plt.tight_layout()
    plt.savefig("AnalyseDataset.png")
    plt.show()




def relabel(label):

    # forward: 0
    # right: 1
    # backward: 2
    # left: 3
    # unknown: 4

    if label == forward_label:
        return 0
    elif label == right_label:
        return 1
    elif label == backward_label:
        return 2
    elif label == left_label:
        return 3
    else:
        return 4

def prepare_data(dataset):

    dataset = dataset.map(lambda wav, label: (wav, relabel(label)))


    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")