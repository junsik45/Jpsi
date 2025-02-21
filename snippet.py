import json
import pandas as pd



index_file = "classification_index.json"
data_file  = "QEC_data.json"

with open(index_file, "r") as f:
    index_data = json.load(f)  # Load the JSON file
# Load Data JSON
with open(data_file, "r") as f:
    data = json.load(f)  # Contains 160,000 keys

# Prepare list to store DataFrame rows
rows = []

# Process each category in index.json
for category, column_indices in index_data.items():
    for col_idx in column_indices:
        col_idx_str = str(col_idx)  # Ensure key is string
        if col_idx_str in data:  # Some indices may be missing
            entry = data[col_idx_str]
            cos_chi = entry[0]
            weight = entry[1]

            # Expand lists into long format
            max_length = max(len(cos_chi), len(weight))  # Get longest list
            for i in range(max_length):
                row = {
                    "category": category,
                    "eventNum": col_idx,
                    "cos_chi_val": cos_chi[i] if i < len(cos_chi) else None,  # Pad if shorter
                    "weight_val": weight[i] if i < len(weight) else None
                }
                rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Display a preview
print(df.head())
df.to_parquet("QEC.parquet", engine="pyarrow", compression="brotli")
