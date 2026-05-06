

import pandas as pd

# === Configuration ===
input_file = "KrPcData_Swi0_35.csv"   # Replace with your actual file name
output_file = "table35.csv"

# === Steps ===
# 1. Load Table 1
df1 = pd.read_csv(input_file)

# 2. Convert Pc to micro units
df1['Pc'] = df1['Pc'] * 1e-6

# 3. Select relevant columns and keep Sw_rev for grouping
df_selected = df1[['Sw_rev', 'Sw', 'kro', 'krw', 'Pc']]

# 4. Sort by Sw ascending within each Sw_rev group
df_sorted = df_selected.sort_values(by=['Sw_rev', 'Sw']).reset_index(drop=True)

# 5. Build CSV string with spaces after commas and empty line when Sw_rev changes
lines = []
prev_sw_rev = None
for _, row in df_sorted.iterrows():
    # Insert empty line if Sw_rev changes
    if prev_sw_rev is not None and row['Sw_rev'] != prev_sw_rev:
        lines.append('')
    # Format row with spaces after commas
    line = f"{row['Sw']}, {row['krw']}, {row['kro']}, {row['Pc']}"
    lines.append(line)
    prev_sw_rev = row['Sw_rev']

# Add header at the top
header = "Sw, kro, krw, Pc"
output_content = header + "\n" + "\n".join(lines)

# Write to file
with open(output_file, 'w') as f:
    f.write(output_content)

print(f"Conversion complete! Table 2 saved as {output_file} with empty lines between Sw_rev groups.")