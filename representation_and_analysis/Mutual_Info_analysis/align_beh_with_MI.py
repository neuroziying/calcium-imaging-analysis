import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

# 加载数据
Neurons = scipy.io.loadmat("C:\\Users\\Administrator\\Desktop\\pycodes\\compact_valid_neurons1011A.mat")  
print("\n type:", type(Neurons))

if 'results' in Neurons:
    results = Neurons['results']
    S_field = results['S'][0, 0]
elif 'compact_neurons' in Neurons:
    results = Neurons['compact_neurons']
    S_field = results['S_signals'][0, 0]
else:
    print("can't find data from mat")
    exit()

spikes = S_field[0,0]
print(f"spikes type: {type(spikes)}")
print(f"spikes shape: {spikes.shape}")

if hasattr(S_field, 'toarray'):
    spikes = S_field.toarray()
else:
    spikes = np.array(S_field)
print(f"arraylike spikes shape: {spikes.shape}")

# 参数设置
total_time = spikes.shape[1]
neuron_num = spikes.shape[0]
sampling_rate = 16
binsize_range = [4,5,6,7]
max_lag = 40


# 时间分箱
right_binsize = 1.2
bi_frames = int(right_binsize * sampling_rate)
num_bins = total_time // bi_frames

# 计算发放率序列 S
S = np.zeros((neuron_num, num_bins))
for i in range(neuron_num):
    for bin_idx in range(num_bins):
        start = bin_idx * bi_frames
        end = start + bi_frames
        S[i, bin_idx] = np.sum(spikes[i, start:end]) / right_binsize

behavior = pd.read_csv(r"C:\Users\Administrator\Desktop\DLC_Projects\MouseBehavior-Ziying_Wang-2025-12-12\analysis_results\mouse_center_trajectory.csv")
frame_series = behavior[0]
time_series = behavior[1]
centerx = behavior[2]
centery = behavior[3]
speed = behavior[4]
centers = [centerx,centery]

grid_len = 4
x_range = np.linspace(np.min(centerx),np.max(centerx),grid_len)
x_range.append(np.max(centerx))
y_range = np.linspace(np.min(centery),np.max(centery),grid_len)
y_range.append(np.max(centery))

def grid(center):
    x,y = center
    for i in range(1,grid_len+1):
        for j in range(1,grid_len+1):
            if x_range[i-1]< x <x_range[i] and y_range[j-1]< y < y_range[j]:
                return i*j

data_matrix = [frame_series,np.zeros()]
for idx,center in enumerate(centers):
    idx = grid(center)


