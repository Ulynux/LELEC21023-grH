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
    with open('classification/data/models/best_rf_model.pickle', 'rb') as file:
        model_rf = pickle.load(file)
        with open('classification/data/models/pca_25_components.pickle', 'rb') as file:
            model_pca = pickle.load(file)

   
            memory = []
            for payload in _input:
                print(f"Payload: {payload}")
                if PRINT_PREFIX in payload:
                    payload = payload[len(PRINT_PREFIX):]

                    melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)
                    logger.info(f"Parsed payload into Mel vectors: {melvec}")

                    melvec = melvec.copy()
                    melvec = melvec.astype(np.float64)

                    melvec -= np.mean(melvec)
                    melvec = melvec / np.linalg.norm(melvec)

                    melvec = melvec.reshape(1, -1)
                    melvec = model_pca.transform(melvec)

                    proba_rf = model_rf.predict_proba(melvec)
                    proba_array = np.array(proba_rf)

                    memory.append(proba_array)

                    # Only predict after 5 inputs
                    if len(memory) >= 5:
                        memory_array = np.array(memory)

                        majority_class_index = np.bincount(np.argmax(memory_array, axis=2).flatten()).argmax()
                        majority_class = model_rf.classes_[majority_class_index]

                        print(f"Majority voting class after 5 inputs: {CLASSNAMES[majority_class]}")

                        memory = []

                        if majority_class == "gun":
                            majority_class = "gunshot"

                        logger.info(f"Predictions: {majority_class}")
                        answer = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{majority_class}", timeout=1)
                        json_answer = json.loads(answer.text)
                        print(json_answer)

                        wait_iterations = np.random.randint(1, 3)
                        for _ in range(wait_iterations):
                            next(_input)