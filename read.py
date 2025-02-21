import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

tensor_dict = {}
data_y = []
df = pd.read_parquet('QEC.parquet', engine='pyarrow')
ev = 40000
df = df.loc[df['eventNum'] < ev ].sort_values(by=['eventNum', 'cos_chi_val'])
print(len(df['category'].unique()))
num_bins = 256
linspace_bin = np.linspace(-1., 1., num_bins+1)
for event, group in df.groupby("eventNum"):
    group["cos_chi_bin"] = pd.cut(group["cos_chi_val"], bins=linspace_bin)
    binned = group.groupby("cos_chi_bin")["weight_val"].sum().reset_index()
    binned["bin_midpoint"] = binned["cos_chi_bin"].apply(lambda x: (x.left + x.right) / 2)

    bin_midpoints_tensor = torch.tensor(binned["bin_midpoint"].values, dtype=torch.float32)
    weight_tensor = torch.tensor(binned["weight_val"].values, dtype=torch.float32)
    #tensor_dict[event] = (bin_midpoints_tensor, weight_tensor)
    tensor_dict[event] = weight_tensor
    data_y.append(event)

data_x = list(tensor_dict.values())
print(data_x)
#ex = df.loc[df['eventNum'] == ev ].sort_values(by='cos_chi_val')
#x = ex['weight_val']
#print( "sum of weights: ", sum(x))
#plt.figure(figsize=(8, 6))
#plt.plot(ex["cos_chi_val"], ex["weight_val"], marker="o", linestyle="")
#num_bins=256
#df["cos_chi_bin"] = pd.cut(ex["cos_chi_val"], bins=np.linspace(-1., 1., num_bins+1))
## Labels and title
#binned = df.groupby("cos_chi_bin")["weight_val"].sum().reset_index()
#binned["bin_midpoint"] = binned["cos_chi_bin"].apply(lambda x: (x.left + x.right) / 2)
#plt.xlabel("cos_chi_val")
#plt.ylabel("weight_val")
#plt.title("Plot of cos_chi_val vs. weight_val")
#plt.grid(True)
#
#plt.savefig('example.pdf')
#plt.clf()
#
#plt.bar(binned['bin_midpoint'], binned['weight_val'], width=0.0075, color='blue')
#plt.xlabel("Binned cos_chi_val")
#plt.ylabel("Mean weight_val")
#plt.savefig('histogram.pdf')
#print(binned)
