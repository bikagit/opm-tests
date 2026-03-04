import pandas as pd

# Filepath to the CSV file
file_path = '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/curated_data/KrPcData_Swi0_06-CORR-OCT2025.csv'

# Load the CSV file
df = pd.read_csv(file_path)

# Add a new column with values equal to 0.06
df['Swi'] = 0.06

# Save the updated CSV file
df.to_csv(file_path, index=False)
