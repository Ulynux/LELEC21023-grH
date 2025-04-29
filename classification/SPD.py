import numpy as np
melvec = [[1] * 400, [1] * 400, [1] * 400, [1] * 400, [10] * 400, [10] * 400]
window_size = 400  # Define the window size for the moving average filter

for i in range(len(melvec)):
    tmp_energy = melvec[i]
    # Apply moving average filter
    moving_avg = np.convolve(tmp_energy, np.ones(window_size) / window_size, mode='valid')
    print(f"Moving average (first 10 values): {moving_avg[0]}")

memory = [
    [0.1, 0.3, 0.5, 0.1],
    [0.2, 0.4, 0.3, 0.1],
    [0.05, 0.25, 0.6, 0.1],
    [0.15, 0.35, 0.4, 0.1],
    [0.1, 0.3, 0.5, 0.1]
]

memory_array = np.array(memory)

majority_class_index = np.bincount(np.argmax(memory_array, axis=1)).argmax()
majority_class = memory_array[majority_class_index]
print(f"Majority voting class: {majority_class}")

## now with log likelihood
memory = [
    [0.30, 0.35, 0.30, 0.05],  # Class 1 and 2 very close
    [0.31, 0.34, 0.30, 0.05],  # Class 1 and 2 very close
    [0.32, 0.33, 0.30, 0.05],  # Class 1 and 2 very close
    [0.30, 0.34, 0.31, 0.05],  # Class 1 and 2 very close
    [0.31, 0.33, 0.31, 0.05]   # Class 1 and 2 very close
]

memory_array = np.array(memory)

# Compute log-likelihoods
log_likelihood = np.log(memory_array)
log_likelihood_sum = np.sum(log_likelihood, axis=0)

# Find the most likely class and the second most likely class
sorted_indices = np.argsort(log_likelihood_sum)[::-1]  # Sort in descending order
most_likely_class_index = sorted_indices[0]
second_most_likely_class_index = sorted_indices[1]

# Compute confidence as the difference between the top two log-likelihoods
confidence = log_likelihood_sum[most_likely_class_index] - log_likelihood_sum[second_most_likely_class_index]

# Define a confidence threshold
confidence_threshold = 0.45  # Adjust this value based on your requirements

if confidence >= confidence_threshold:
    print(f"Most likely class index: {most_likely_class_index}")
    print(f"Confidence: {confidence}")
    print("Submitting the guess...")
else:
    print(f"Confidence too low ({confidence}). Not submitting the guess.")

a = np.ones((20,20))
a = a.reshape(-1)
print(a)
print("aaaaaaaaaaaaaaaaaaaaa",a.shape)


b = a.reshape(-1,20,20,1)
c = a.reshape(20,20,1)
print(b.shape,c.shape)

import random  # <-- add this!
import pandas as pd
import numpy as np

# Fake Grid Search Results
np.random.seed(42)
random.seed(42)  # optional, to match numpy randomness
param_combinations = 30

results_df = pd.DataFrame({
    'param_model__kernel_size': random.choices([(3,3), (5,5)], k=param_combinations),
    'param_model__activation': random.choices(['relu', 'tanh'], k=param_combinations),
    'param_model__optimizer': random.choices(['adam', 'sgd', 'rmsprop'], k=param_combinations),
    'param_model__learning_rate': random.choices([0.001, 0.01], k=param_combinations),
    'mean_test_score': np.random.uniform(0.7, 0.95, param_combinations)
})
