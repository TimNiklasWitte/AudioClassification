import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():

    labels = ["top", "right", "down", "left", "unknown"]
    alpha_range = list(np.arange(1, 0.0, -0.1)) + list(np.arange(0.1, 0.00, -0.01))

    dfs = []
    for noise_type in ["idel", "walk"]:
        for alpha in alpha_range:
            file_name = f"./relative numpy/{noise_type}/ConfusionMatrix_{alpha:.2f}.npy"
            confusion_matrix = np.load(file_name)

            diag = np.diag(confusion_matrix)
            
            for idx, label in enumerate(labels):
                data = {
                    "Noise type": noise_type,
                    "Alpha": alpha,
                    "Label": label,
                    "Accuracy": diag[idx]
                }

                df = pd.DataFrame(data, index=[0])
                dfs.append(df)

    df = pd.concat(dfs)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))


    sns.set_style("ticks",{'axes.grid' : True})

    df_idel = df.loc[df["Noise type"] == "idel"]
    df_walk = df.loc[df["Noise type"] == "walk"]
 
    lineplot = sns.lineplot(df_idel, x="Alpha", y ="Accuracy", hue="Label", ax=axs[0])
    axs[0].set_title("Idel")
    axs[0].grid()
    axs[0].axvline(x=0.01, c="black")
    axs[0].axvline(x=0.5, c="black")

    lineplot.invert_xaxis()

    lineplot = sns.lineplot(df_walk, x="Alpha", y ="Accuracy", hue="Label", ax=axs[1])
    
    axs[1].set_title("Walk")
    axs[1].grid()
     
    axs[1].axvline(x=0.4, c="black")
    axs[1].axvline(x=1, c="black")

    lineplot.invert_xaxis()

    plt.tight_layout()
    plt.savefig("Overview.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")