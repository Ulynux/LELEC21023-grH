import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import click
import matplotlib.pyplot as plt
import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
import keras
import requests
import json
from .utils import payload_to_melvecs
from collections import deque
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
    filename="specgram_plot.pdf"
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
    plt.savefig(filename, format='pdf')
    return None
hostname = "http://lelec210x.sipr.ucl.ac.be"
key = "t-meiXk3RHRYwfdeGoM8fObRRnggVsjrv6KToE5r" #################################
load_dotenv()

CLASSNAMES = ["chainsaw", "fire", "fireworks","gunshot"]

@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model: Optional[Path],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).

    Most likely, you want to pipe this script after running authentification
    on the packets:

        rye run auth | rye run classify

    This way, you will directly receive the authentified packets from STDIN
    (standard input, i.e., the terminal).
    """
    model = keras.models.load_model('classification/data/models/CNN_good_bcr.keras')


    moving_avg = 0
    threshold = 3.5
    energy_flag = False
    memory = []
    ## Using a queue to store avg of before 
    i = 0
    long_sum = deque(maxlen=10)
    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX):]

            melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)
            melvec = melvec.copy().astype(np.float64)

            # Compute the energy
            short_sum_1 = np.convolve(melvec.reshape(-1)[:200], np.ones(200) / 200, mode='valid')[0]
            short_sum_2 = np.convolve(melvec.reshape(-1)[200:], np.ones(200) / 200, mode='valid')[0]

            
            if moving_avg == 0:
                moving_avg = short_sum_1  
                          
            if short_sum_1 >= threshold * moving_avg :
                energy_flag = True
            else:
                long_sum.append(short_sum_1)
                moving_avg = np.mean(long_sum)
            
            
            if not energy_flag:
                if short_sum_2 >= threshold * moving_avg  :
                    energy_flag = True
                else:
                    long_sum.append(short_sum_2)

                    moving_avg = np.mean(long_sum)
                    logger.info(f"moving_avg  : {moving_avg.round(5)}")

            if energy_flag: # Mtn que l'on est sur qu'il y a un signal, on peut faire la classification 
                            # sans regarder à la valeur du moving average car on ne va pas regarder 
                            # qu'après on a plus de signal et stopper la classif en plein milieu
                            # de celle-ci et recommencer à chaque fois 
                        
                logger.info(f"Starting classification")
                
                melvec -= np.mean(melvec)
                melvec = melvec / np.linalg.norm(melvec)

                melvec = melvec.reshape((20, 20)).T
                melvec = melvec.reshape((-1, 20, 20, 1))
                
                fig, ax = plt.subplots(figsize=(3, 3))  
                plot_specgram(
                    melvec.reshape((20, 20)).T,  # Reshape le melvec pour l'affichage
                    ax=ax,
                    is_mel=True,
                    title=f"Mel {i}",  
                    xlabel="Mel vector"
                )
                fig.savefig(f"mcu/hands_on_feature_vectors/mel_{i}.png")  # Sauvegardez le plot
                plt.close(fig)
                i += 1
                proba = model.predict(melvec)
                proba_array = np.array(proba)

                memory.append(proba_array)

                # Only predict after 5 inputs
                if len(memory) >= 5:
                    
                    
                    memory_array = np.array(memory)
                    
                    ## going from shape (5,1,4) to (5,4)

                    memory_array = memory_array.reshape(memory_array.shape[0], -1)
                    logger.info(memory_array)

                    log_likelihood = np.log(memory_array + 1e-10)  # Adding a small value to avoid log(0)
                    log_likelihood_sum = np.sum(log_likelihood, axis=0)

                    sorted_indices = np.argsort(log_likelihood_sum)[::-1]  # Sort in descending order
                    most_likely_class_index = sorted_indices[0]
                    # On revient à un état où on relance la classification depuis le début
                    # => on clear la mémoire, et on relance le moving average mais on garde les valeurs 
                    # du moving average précédent sinon on perds trop d'infos
                                                
                    energy_flag  = False
                    memory = []
                    
                    
                    majority_class = CLASSNAMES[most_likely_class_index]
                    if majority_class == "gun":
                        majority_class = "gunshot"
                    logger.info(f"Most likely class index: {majority_class}")
            