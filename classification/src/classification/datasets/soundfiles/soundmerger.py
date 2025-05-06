import os
import soundfile as sf
import numpy as np
from collections import defaultdict
from pathlib import Path

def merge_sounds(file_paths, output_path):
    """
    Fusionne les fichiers audio spécifiés dans file_paths et sauvegarde le résultat dans output_path.
    """
    merged_data = []
    sample_rate = None

    for file_path in file_paths:
        data, fs = sf.read(file_path)
        if sample_rate is None:
            sample_rate = fs
        elif sample_rate != fs:
            raise ValueError(f"Sample rate mismatch: {file_path} has {fs}, expected {sample_rate}")
        merged_data.append(data)

    # Concaténer les données audio
    merged_data = np.concatenate(merged_data, axis=0)
    sf.write(output_path, merged_data, sample_rate)
    print(f"Fichier fusionné sauvegardé : {output_path}")

# Chemin vers le dossier contenant les fichiers audio
base_dir = Path("classification/src/classification/datasets/soundfiles_old/")
output_dir = Path("classification/src/classification/datasets/soundfiles/")

# Créer le dossier de sortie s'il n'existe pas
output_dir.mkdir(parents=True, exist_ok=True)

# Classes à traiter et leurs séparateurs
classes_to_merge = {
    "fire": "_",
    "chainsaw": "_",
    "fireworks": "-",
    "gun": "-"
}

for class_name, separator in classes_to_merge.items():
    print(f"Traitement de la classe : {class_name}")
    class_files = sorted([f for f in base_dir.glob(f"{class_name}*") if f.suffix == ".wav"])

    if not class_files:
        print(f"Aucun fichier trouvé pour la classe {class_name}.")
        continue

    # Regrouper les fichiers par paires
    grouped_files = defaultdict(list)
    for file_path in class_files:
        # Extraire l'identifiant principal
        if class_name == "gun":
            # Pour la classe gun, regrouper par tranche de 20 (ex. gun-000 et gun-010 -> gun-00)
            identifier = separator.join(file_path.stem.split(separator)[:-1])
            pair_index = int(file_path.stem.split(separator)[-1]) // 20  # Diviser par 20 pour regrouper
            grouped_files[f"{identifier}{separator}{pair_index:02d}"].append(file_path)
        else:
            # Pour les autres classes, regrouper par les deux premiers chiffres
            identifier = separator.join(file_path.stem.split(separator)[:-1])
            pair_index = file_path.stem.split(separator)[-1][:-1]  # Garder les deux premiers chiffres
            grouped_files[f"{identifier}{separator}{pair_index}"].append(file_path)

    # Fusionner les fichiers pour chaque paire
    for identifier, file_paths in grouped_files.items():
        if len(file_paths) == 2:  # Fusionner uniquement s'il y a exactement deux fichiers
            output_file = output_dir / f"{identifier}.wav"
            merge_sounds(file_paths, output_file)

print("Traitement des cas particuliers pour fireworks")
path1 = "classification/src/classification/datasets/soundfiles_old/fireworks-000.wav"
path2 = "classification/src/classification/datasets/soundfiles_old/fireworks-010.wav"
path3 = "classification/src/classification/datasets/soundfiles_old/fireworks-020.wav"
path4 = "classification/src/classification/datasets/soundfiles_old/fireworks-030.wav"
path5 = "classification/src/classification/datasets/soundfiles_old/fireworks-040.wav"

# Fusionner les fichiers spécifiques
merge_sounds([path1, path2], "classification/src/classification/datasets/soundfiles/fireworks-00.wav")
merge_sounds([path3, path4], "classification/src/classification/datasets/soundfiles/fireworks-01.wav")
merge_sounds([path5, path1], "classification/src/classification/datasets/soundfiles/fireworks-02.wav")




