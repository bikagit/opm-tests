# import pandas as pd

# # Filepath to the CSV file
# file_path = '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_80.csv'

# # Load the CSV file
# df = pd.read_csv(file_path)

# # Add a new column with values equal to 0.06
# df['Swi'] = 0.80

# # Save the updated CSV file
# df.to_csv(file_path, index=False)

# print("Column added successfully.")

# import pandas as pd
# import glob

# # Define the file paths for all the CSV files
# file_paths = [
#     '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_06.csv',
#     '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_20.csv',
#     '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_35.csv',
#     '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_50.csv',
#     '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_65.csv',
#     '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_80.csv'

# ]

# # Read and combine all CSV files into one DataFrame
# combined_df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# # Save the combined DataFrame to a new CSV file
# output_file = '/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Combinedtot.csv'
# combined_df.to_csv(output_file, index=False)

# print(f"Combined file saved to: {output_file}")

# import pandas as pd

# # Load the CSV file into a DataFrame
# df = pd.read_csv('/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst/data/KrPcData_Swi0_20.csv')

# # List the columns you want to remove
# columns_to_remove = ['Process', 'Rev_num','Sw_rev','Swi']

# # Drop the specified columns
# df.drop(columns=columns_to_remove, inplace=True)

# # Save the updated DataFrame back to a CSV file
# df.to_csv('your_file_updated.csv', index=False)

# print("Columns removed and file saved successfully.")





import pandas as pd

# Load the CSV file into a DataFrame using space as the delimiter
# Load the CSV file into a DataFrame
df = pd.read_csv('KrPcData_Combined080.csv')

# Define the new order of columns
new_order = ['Sw', 'krw', 'kro','Pc']  # Adjust this list as needed

# Reorder the DataFrame columns
df = df[new_order]

# Save the updated DataFrame back to a CSV file using space as the delimiter
df.to_csv('your_file_updated.csv', sep=' ', index=False)

print("Columns switched and file saved successfully with spaces as delimiters.")
# import pandas as pd

# Load the CSV file into a DataFrame using space as the delimiter
df = pd.read_csv('your_file_updated.csv')

# # Reverse the order of rows
# df = df.iloc[::-1]

# Save the updated DataFrame back to a CSV file using space as the delimiter
df.to_csv('your_file_updated.csv', sep=' ', index=False)

print("Rows reversed and file saved successfully.")


# import pandas as pd
import csv

# # Load the CSV file into a DataFrame using space as the delimiter
# df = pd.read_csv('your_file_updated.csv', delimiter=' ')

# # Remove double quotes from all columns
# df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

# # Save the updated DataFrame back to a CSV file using space as the delimiter
# df.to_csv('your_file_updated.csv', sep=' ', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

# print("Double quotes removed and file saved successfully.")


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('KrPcData_Combined080.csv')

# Define the new order of columns
new_order = ['Sw', 'krw', 'kro', 'Pc']  # Adjust this list as needed

# Reorder the DataFrame columns
df = df[new_order]

# Remove double quotes from all columns
df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

# Save the updated DataFrame back to a CSV file using space as the delimiter
df.to_csv('your_file_updated.csv', sep=' ', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

print("Columns switched, double quotes removed, and file saved successfully with spaces as delimiters.")

# Load the updated CSV file into a DataFrame using space as the delimiter
df = pd.read_csv('your_file_updated.csv', delimiter=' ')

# Reverse the order of rows
# df = df.iloc[::-1]

# Save the updated DataFrame back to a CSV file using space as the delimiter
df.to_csv('your_file_updated.csv', sep=' ', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

print("Rows reversed and file saved successfully.")
