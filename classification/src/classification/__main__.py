import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import click

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger

import requests
import json
from .utils import payload_to_melvecs
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
    with open('classification/data/models/best_GB_model.pickle', 'rb') as file:
        model_rf = pickle.load(file)
        with open('classification/data/models/pca_29_GB_components.pickle', 'rb') as file:
            model_pca = pickle.load(file)

            moving_avg = 0
            threshold = 5
            energy_flag = False
            memory = []
            
            for payload in _input:
                if PRINT_PREFIX in payload:
                    payload = payload[len(PRINT_PREFIX):]

                    melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)
                    melvec = melvec.copy()
                    melvec = melvec.astype(np.float64)

                    # melvec is a 20x20 array => compute the energy to see if there is a signal
                    tmp_moving_avg = np.convolve(melvec.reshape(-1), np.ones(400) / 400, mode='valid')[0] 
                    # convolve retourne un tableau de 1 valeur donc on prend le premier élément

                    if moving_avg == 0:
                        moving_avg = tmp_moving_avg

                    # Threshold ajustable mais 5 * le moving average parait OK
                    if tmp_moving_avg > threshold * moving_avg:
                        energy_flag = True
                        logger.info(f"Energy spike detected. Threshold: {threshold * moving_avg}")
                    else:
                        # moyenne des 2 moving average
                        moving_avg = (moving_avg + tmp_moving_avg) / 2

                    if energy_flag: # Mtn que l'on est sur qu'il y a un signal, on peut faire la classification 
                                    # sans regarder à la valeur du moving average car on ne va pas regarder 
                                    # qu'après on a plus de signal et stopper la classif en plein milieu
                                    # de celle-ci et recommencer à chaque fois 
                                
                        logger.info(f"Starting classification")
                        
                        melvec -= np.mean(melvec)
                        melvec = melvec / np.linalg.norm(melvec)

                        melvec = melvec.reshape(-1)
                        melvec = model_pca.transform(melvec)

                        proba_rf = model_rf.predict_proba(melvec)
                        proba_array = np.array(proba_rf)

                        memory.append(proba_array)

                        # Only predict after 5 inputs
                        if len(memory) >= 5:
                            
                            
                            memory_array = np.array(memory)

                            log_likelihood = np.log(memory_array)
                            log_likelihood_sum = np.sum(log_likelihood, axis=0)

                            sorted_indices = np.argsort(log_likelihood_sum)[::-1]  # Sort in descending order
                            most_likely_class_index = sorted_indices[0]
                            second_most_likely_class_index = sorted_indices[1]

                            confidence = log_likelihood_sum[most_likely_class_index] - log_likelihood_sum[second_most_likely_class_index]

                            # threshold sur la confiance de la prédiction
                            
                            confidence_threshold = 0.45  
                            print(f"Majority voting class after 5 inputs: {majority_class}")

                            # On revient à un état où on relance la classification depuis le début
                            # => on clear la mémoire, et on relance le moving average mais on garde les valeurs 
                            # du moving average précédent sinon on perds trop d'infos
                                                       
                            energy_flag  = False
                            memory = []
                            
                            if majority_class == "gun":
                                majority_class = "gunshot"
                            majority_class = CLASSNAMES[majority_class-1]
                            
                            if confidence >= confidence_threshold:
                                logger.info(f"Most likely class index: {most_likely_class_index}")
                                logger.info(f"Confidence: {confidence}")
                                answer = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{majority_class}", timeout=1)
                                json_answer = json.loads(answer.text)
                                print(json_answer)                            
                            else:
                                logger.info(f"Confidence too low ({confidence}). Not submitting the guess.")
                          