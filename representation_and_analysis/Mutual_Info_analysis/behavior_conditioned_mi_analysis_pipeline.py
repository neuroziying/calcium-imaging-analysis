# ===============================
# Behavior-conditioned neural MI analysis pipeline
# Author: ChatGPT (skeleton)
# Purpose: DLC position + neural activity -> spatial bins -> MI / lagged MI / clustering
# ===============================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from scipy.stats import zscore
import scipy.io
import pandas as pd
# -------------------------------------------------
# 0. Helper functions
# -------------------------------------------------

def discretize_position(x, y, n_bins=4):
    """
    Discretize 2D position into n_bins x n_bins spatial bins.
    Returns:
        z(t): integer bin index in [0, n_bins^2-1]
    """
    x_bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    y_bins = np.linspace(np.min(y), np.max(y), n_bins + 1)

    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1

    x_idx = np.clip(x_idx, 0, n_bins - 1)
    y_idx = np.clip(y_idx, 0, n_bins - 1)

    z = y_idx * n_bins + x_idx
    return z


def make_shared_bins(S, num_bins=8):
    """Shared histogram bins for MI."""
    s_min, s_max = np.min(S), np.max(S)
    return np.linspace(s_min, s_max, num_bins + 1)


def mutual_information(x, y, bins):
    """Basic histogram MI."""
    px, _ = np.histogram(x, bins=bins, density=True)
    py, _ = np.histogram(y, bins=bins, density=True)
    pxy, _, _ = np.histogram2d(x, y, bins=[bins, bins], density=True)

    px += 1e-12
    py += 1e-12
    pxy += 1e-12

    mi = np.sum(pxy * np.log(pxy / (px[:, None] * py[None, :])))
    return mi


def lagged_mi(x, y, bins, lag):
    if lag > 0:
        return mutual_information(x[:-lag], y[lag:], bins)
    elif lag < 0:
        return mutual_information(x[-lag:], y[:len(y)+lag], bins)
    else:
        return mutual_information(x, y, bins)


# -------------------------------------------------
# 1. Load data (replace with your real loading code)
# -------------------------------------------------
# Neural data: S shape = (N_neurons, T)
# Position data: x(t), y(t)
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
total_time = spikes.shape[1]
neuron_num = spikes.shape[0]
sampling_rate = 16
max_lag = 40


# 时间分箱
right_binsize = 1.2  #秒
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

frame_series = behavior.iloc[:, 0].values
time_series  = behavior.iloc[:, 1].values
x = behavior.iloc[:, 2].values
y = behavior.iloc[:, 3].values
speed = behavior.iloc[:, 4].values

# 将行为轴与时间轴对齐

t_beh = np.arange(len(x)) / 30.0            # behavior time (s)
t_neu = np.arange(spikes.shape[1]) / 16.67  # neural time (s)

# 插值
x_neu = np.interp(t_neu, t_beh, x)
y_neu = np.interp(t_neu, t_beh, y)



# -------------------------------------------------
# 2. Spatial binning (aligned to neural bins)
# -------------------------------------------------

n_bins = 4
num_states = n_bins ** 2

# discretize at neural-frame resolution
z_neu = discretize_position(x_neu, y_neu, n_bins=n_bins)

# collapse to neural bins (1.2 s)
z_bin = np.zeros(num_bins, dtype=int)

for b in range(num_bins):
    start = b * bi_frames
    end = start + bi_frames
    z_bin[b] = np.bincount(z_neu[start:end]).argmax()


# -------------------------------------------------
# 3. Build state-conditioned neural datasets
# -------------------------------------------------

N, T = S.shape
state_indices = [np.where(z_bin == k)[0] for k in range(num_states)]

# Optional: remove states with too little occupancy
min_samples = 5
valid_states = [k for k in range(num_states) if len(state_indices[k]) > min_samples]
if len(valid_states) == 0:
    print("No valid states, please try lowering min_samples")
    exit()

plt.figure(figsize=(8,2))
plt.plot(z_bin, '.')
plt.xlabel("Neural bin")
plt.ylabel("Behavior state")
plt.title("Behavior state per neural bin")
plt.show()

unique, counts = np.unique(z_bin, return_counts=True)
print(dict(zip(unique, counts)))

print("Valid states:", valid_states)
print("Occupancy:", [len(state_indices[k]) for k in valid_states])
# -------------------------------------------------
# 4. Mean activity per state
# -------------------------------------------------

mean_activity = np.zeros((num_states, N))
for k in valid_states:
    mean_activity[k] = np.mean(S[:, state_indices[k]], axis=1)

# -------------------------------------------------
# 5. MI matrix per state
# -------------------------------------------------

bins = make_shared_bins(S, num_bins=8)
MI_state = {}

for k in valid_states:
    idx = state_indices[k]
    MI = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            mi = mutual_information(S[i, idx], S[j, idx], bins)
            MI[i, j] = MI[j, i] = mi
    MI_state[k] = MI

# # -------------------------------------------------
# # 6. (Optional) Lagged MI per state
# # -------------------------------------------------

# lags = np.arange(-10, 11)
# MI_lag_state = {}

# for k in valid_states:
#     idx = state_indices[k]
#     MI_lag = np.zeros((N, N, len(lags)))
#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 continue
#             for t, lag in enumerate(lags):
#                 MI_lag[i, j, t] = lagged_mi(S[i, idx], S[j, idx], bins, lag)
#     MI_lag_state[k] = MI_lag

# -------------------------------------------------
# 6. Lagged MI (global, toy model)
# -------------------------------------------------

# lags = np.arange(-10, 11)
# MI_lag = np.zeros((N, N, len(lags)))

# for i in range(N):
#     for j in range(N):
#         if i == j:
#             continue
#         for t, lag in enumerate(lags):
#             MI_lag[i, j, t] = lagged_mi(S[i], S[j], bins, lag)


# -------------------------------------------------
# 7. Simple clustering example (per state)
# -------------------------------------------------
# # NOTE: exploratory clustering (placeholder, no strong interpretation yet)

# clusters = {}
# for k in valid_states:
#     MI = MI_state[k]
#     affinity = MI / np.max(MI)
#     model = SpectralClustering(
#         n_clusters=4,
#         affinity='precomputed',
#         assign_labels='kmeans'
#     )
#     labels = model.fit_predict(affinity)
#     clusters[k] = labels

# -------------------------------------------------
# 8. Visualization examples
# -------------------------------------------------
# for k in valid_states: 
#     plt.figure(figsize=(10, 10)) 
#     plt.imshow(MI_state[k], cmap='viridis') 
#     plt.colorbar(label='MI') plt.title(f'MI matrix (state {k})')
#     plt.tight_layout() 
#     output_path = rf'C:\Users\Administrator\Desktop\mutual info\1011A\behavior\state{k}.png' 
#     plt.savefig(output_path, dpi=150, bbox_inches='tight') 
#     plt.show() 
    
# for k in valid_states: 
#     plt.figure(figsize=(10, 6)) 
#     plt.bar(np.arange(N), mean_activity[k]) 
#     plt.title(f'Mean firing per neuron (state {k})') 
#     plt.xlabel('Neuron') 
#     plt.ylabel('Activity') 
#     plt.tight_layout() 
#     output_path = rf'C:\Users\Administrator\Desktop\mutual info\1011A\behavior\firing{k}.png' 
#     plt.savefig(output_path, dpi=150, bbox_inches='tight') 
#     plt.show()



# -------------------------------------------------
# 9. Visualization examples  GLOBAL
# -------------------------------------------------


# global_max = max(np.max(MI_state[k]) for k in valid_states)

# for k in valid_states:
#     plt.figure(figsize=(10, 10))
#     plt.imshow(
#         MI_state[k],
#         cmap='viridis',
#         vmin=0,
#         vmax=global_max
#     )
#     plt.colorbar(label='MI')
#     plt.title(f'MI matrix (state {k})')
#     plt.tight_layout()

#     output_path = rf'C:\Users\Administrator\Desktop\mutual info\1011A\behavior\state{k}.png'
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.show()

# global_ymax = max(np.max(mean_activity[k]) for k in valid_states)

# for k in valid_states:
#     plt.figure(figsize=(10, 6))
#     plt.bar(np.arange(N), mean_activity[k])
#     plt.ylim(0, global_ymax)

#     plt.title(f'Mean firing per neuron (state {k})')
#     plt.xlabel('Neuron')
#     plt.ylabel('Activity')
#     plt.tight_layout()

#     output_path = rf'C:\Users\Administrator\Desktop\mutual info\1011A\behavior\firing{k}.png'
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.show()

#==========================================================
import numpy as np

num_states = 16

# 每个神经元的 16-state rate map
rate_maps = np.full((neuron_num, num_states), np.nan)

for k in valid_states:
    rate_maps[:, k] = mean_activity[k]

# 空间 selectivity 指标
# SI_i = max_k r_i(k) / mean_k r_i(k)
selectivity_neuron = np.nanmax(rate_maps, axis=1) / (
    np.nanmean(rate_maps, axis=1) + 1e-12
)

# 排序
sorted_idx = np.argsort(selectivity_neuron)[::-1]

# 选前 10%
top_fraction = 0.1
n_top = int(neuron_num * top_fraction)
top_neurons = sorted_idx[:n_top]

print(f"Top {n_top} spatially selective neurons:")
print(top_neurons)


for i in top_neurons:
    rate_map = np.full(16, np.nan)

    for k in valid_states:
        rate_map[k] = mean_activity[k, i]

    heat_rate_map = rate_map.reshape(4, 4)

    plt.figure(figsize=(6, 6))
    plt.imshow(heat_rate_map, cmap='hot')
    plt.colorbar(label='Firing rate')
    plt.title(f'Neuron {i} spatial tuning')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 用 nan 填 0（toy model 阶段完全 OK）
X = np.nan_to_num(rate_maps)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.6)
plt.scatter(
    X_pca[top_neurons, 0],
    X_pca[top_neurons, 1],
    color='red',
    s=15,
    label='Top selective neurons'
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of spatial rate maps')
plt.legend()
plt.tight_layout()
plt.show()

import umap

reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.3,
    random_state=0
)

X_umap = reducer.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=10, alpha=0.6)
plt.scatter(
    X_umap[top_neurons, 0],
    X_umap[top_neurons, 1],
    color='red',
    s=15,
    label='Top selective neurons'
)
plt.title('UMAP of spatial rate maps')
plt.legend()
plt.tight_layout()
plt.show()

state_k = 10

firing_k = mean_activity[state_k]   # shape: (N,)
sorted_neurons = np.argsort(firing_k)[::-1]

top_k = 10
top_neurons_state = sorted_neurons[:top_k]

print(f"Top {top_k} neurons in state {state_k}:")
print(top_neurons_state)

for i in top_neurons_state:
    rate_map = np.full(16, np.nan)
    for k in valid_states:
        rate_map[k] = mean_activity[k, i]

    heatmap = rate_map.reshape(4, 4)

    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar(label='Firing rate')
    plt.title(f'Neuron {i}')
    plt.tight_layout()
    plt.show()

def compute_selectivity(rate_map):
    return np.nanmax(rate_map) / (np.nanmean(rate_map) + 1e-12)

real_selectivity = np.array([
    compute_selectivity(rate_maps[i])
    for i in range(neuron_num)
])

n_shuffle = 1000
selectivity_null = np.zeros((neuron_num, n_shuffle))

rng = np.random.default_rng(seed=0)

for i in range(neuron_num):
    r = rate_maps[i].copy()

    # 去掉 nan（只在 valid states shuffle）
    valid_idx = ~np.isnan(r)
    r_valid = r[valid_idx]

    for s in range(n_shuffle):
        shuffled = rng.permutation(r_valid)
        r_shuff = r.copy()
        r_shuff[valid_idx] = shuffled

        selectivity_null[i, s] = compute_selectivity(r_shuff)

p_values = np.zeros(neuron_num)

for i in range(neuron_num):
    p_values[i] = np.mean(selectivity_null[i] >= real_selectivity[i])

alpha = 0.05
significant_neurons = np.where(p_values < alpha)[0]

print(f"Significant spatial neurons: {len(significant_neurons)}")
print(significant_neurons)

i = significant_neurons[0]

plt.figure(figsize=(5, 4))
x = np.asarray(selectivity_null[i]).astype(float).ravel()

plt.figure()

if len(x) < 2 or np.nanstd(x) < 1e-12:
    # 极端情况：数据太少 / 几乎没有分布
    plt.bar([0], [len(x)], width=0.5, alpha=0.7, label='Null')
else:
    bins = min(20, len(np.unique(x)))
    plt.hist(x, bins=bins, alpha=0.7, label='Null')

plt.axvline(real_selectivity[i], color='r', linewidth=2, label='Real')
plt.xlabel('Selectivity')
plt.ylabel('Count')
plt.title(f'Neuron {i}')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.hist(p_values, bins='auto')
plt.xlabel('p-value')
plt.ylabel('Count')
plt.title('P-value distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.4)
plt.scatter(
    X_pca[significant_neurons, 0],
    X_pca[significant_neurons, 1],
    color='red',
    s=20,
    label='Significant spatial neurons'
)
plt.legend()
plt.title('PCA with spatially selective neurons')
plt.tight_layout()
plt.show()
