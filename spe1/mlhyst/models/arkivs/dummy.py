from sklearn.preprocessing import MinMaxScaler

import numpy as np

# Example data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Initialize and fit the scaler
scaler = MinMaxScaler()
scaler.fit(data)

# Transform the data
rescaled_data = scaler.transform(data)

# Print the outputs
print("Original Data:\n", data)
print("Rescaled Data (0 to 1):\n", rescaled_data)
