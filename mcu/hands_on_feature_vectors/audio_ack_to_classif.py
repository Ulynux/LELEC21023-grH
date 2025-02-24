import numpy as np

# Load the melspectrograms.npy file
all_classes = ["gunshot", "fireworks", "chainsaw", "crackling fire"]
classe = all_classes[0]
melspectrograms = np.load("mcu/hands_on_feature_vectors/melspectrograms"+str(classe)+".npy")
labels = np.load("mcu/hands_on_feature_vectors/labels"+str(classe)+".npy")
print(labels,labels.shape)
print(melspectrograms)
for i in range(0, len(labels)):
    print(labels[i])
    print(melspectrograms[i])
    print("=====================================")
# Print the shape of the loaded data
print(melspectrograms.shape)