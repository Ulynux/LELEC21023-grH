import numpy as np

# Load the melspectrograms.npy file
all_classes = ["gunshot", "fireworks", "chainsaw", "crackling fire"]
classe = all_classes[0]
melspectrograms = np.load("mcu/hands_on_main_app/melspectrograms"+str(classe)+".npy")
labels = np.load("mcu/hands_on_main_app/labels"+str(classe)+".npy")
print(labels,labels.shape)
print(melspectrograms)
# Print the shape of the loaded data
print(melspectrograms.shape)