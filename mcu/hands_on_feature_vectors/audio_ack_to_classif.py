import numpy as np

# Load the melspectrograms.npy file
all_classes = ["gun", "fireworks", "chainsaw", "fire"]
classe = all_classes[2]
melspectrograms = np.load("mcu/hands_on_feature_vectors/melspectrograms"+str(classe)+".npy")
labels = np.load("mcu/hands_on_feature_vectors/labels"+str(classe)+".npy")
print(labels,labels.shape)
melspectrograms = melspectrograms[:-1]
labels = labels[:-1]
for i in range(0, len(labels)):
    print(labels[i])
    print(melspectrograms[i])
    print("=====================================")
# Print the shape of the loaded data
print(melspectrograms.shape)

def plot_specgram(
    specgram,
    ax,
    is_mel=False,
    title=None,
    xlabel="Time [s]",
    ylabel="Frequency [Hz]",
    cmap="jet",
    cb=True,
    tf=None,
    invert=True,
):
    """
    Plot a spectrogram (2D matrix) in a chosen axis of a figure.
    Inputs:
        - specgram = spectrogram (2D array)
        - ax       = current axis in figure
        - title
        - xlabel
        - ylabel
        - cmap
        - cb       = show colorbar if True
        - tf       = final time in xaxis of specgram
    """
    if tf is None:
        tf = specgram.shape[1]

    if is_mel:
        ylabel = "Frequency [Mel]"
        im = ax.imshow(
            specgram, cmap=cmap, aspect="auto", extent=[0, tf, specgram.shape[0], 0]
        )
    else:
        im = ax.imshow(
            specgram,
            cmap=cmap,
            aspect="auto",
            extent=[0, tf, int(specgram.size / tf), 0],
        )
    if invert:
        ax.invert_yaxis()
    fig = plt.gcf()
    if cb:
        fig.colorbar(im, ax=ax)
    # cbar.set_label('log scale', rotation=270)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return None
import matplotlib.pyplot as plt
# Plot the spectrogram
N_MELVECS, MELVEC_LENGTH = 20,20

for i in range(len(melspectrograms)):
    melvec = melspectrograms[i]  # Plot each melspectrogram

    fig, ax = plt.subplots()
    plot_specgram(
        melvec.reshape((20, 20)).T,
        ax=ax,
        is_mel=True,
        title=f"Mel Spectrogram of {classe} - Sample {i}",
        xlabel="Time [s]",
        ylabel="Frequency [Mel]"
    )
    plt.savefig(f"mcu/hands_on_feature_vectors/plotchainsaw/mel_spectrogram_{classe}_{i}.png")
    plt.close(fig)
