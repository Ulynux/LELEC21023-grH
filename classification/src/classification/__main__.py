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
hostname = "http://localhost:5000"
key = "TOADD" #################################
load_dotenv()


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
    with open('classification/data/models/modeltest.pickle', 'rb') as file:
        model_knn = pickle.load(file)
    with open('classification/data/models/pca.pickle', 'rb') as file:
        model_pca = pickle.load(file)
   

    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvecs}")

            if model_knn and model_pca:
                melvec = melvec/np.linalg.norm(melvec)
                melvec = melvec.reshape(1, -1)
                melvec_reduced = model_pca.transform(melvec)
                proba_knn = model_knn.predict_proba(melvec_reduced)
                prediction = model_knn.predict(melvec_reduced)
                
                logger.info(f"Predictions: {proba_knn}")
                
                answer = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{prediction}", timeout=1)
                
                json_answer = json.loads(answer.text)
                print(json_answer)
