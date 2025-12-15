import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf


def make_shared_bins(data, num_bins):
    """
    data: (neurons, time)
    """
    vmin = np.min(data)
    vmax = np.max(data)
    return np.linspace(vmin, vmax, num_bins + 1)



def entropy_discrete(x, bins):
    counts, _ = np.histogram(x, bins=bins)
    p = counts / np.sum(counts)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))



def mutual_information_discrete(x, y, bins):
    joint_counts, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    p_xy = joint_counts / np.sum(joint_counts)

    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(
                    p_xy[i, j] / (p_x[i] * p_y[j])
                )
    return mi



def compute_mi_matrix(data, num_bins=8):
    n = data.shape[0]
    bins = make_shared_bins(data, num_bins)

    MI = np.zeros((n, n))
    H = np.zeros(n)

    for i in range(n):
        H[i] = entropy_discrete(data[i], bins)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                MI[i, j] = H[i]
            else:
                mi = mutual_information_discrete(
                    data[i], data[j], bins
                )
                MI[i, j] = mi
                MI[j, i] = mi

    return MI, H

def compute_nmi_dmi(MI, H):
    n = len(H)
    NMI = np.zeros((n, n))
    DMI = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if H[i] + H[j] > 0:
                NMI[i, j] = 2 * MI[i, j] / (H[i] + H[j])
            if H[i] > 0:
                DMI[i, j] = MI[i, j] / H[i]
    return NMI, DMI

def lagged_mi(x, y, bins, lag):
    """
    lag > 0: x_t vs y_{t+lag}
    """
    if lag > 0:
        return mutual_information_discrete(
            x[:-lag], y[lag:], bins
        )
    elif lag < 0:
        return mutual_information_discrete(
            x[-lag:], y[:lag], bins
        )
    else:
        return mutual_information_discrete(x, y, bins)
    


def compute_lagged_mi_matrix(data, lag, num_bins=8):
    n = data.shape[0]
    bins = make_shared_bins(data, num_bins)
    MI_lag = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            MI_lag[i, j] = lagged_mi(
                data[i], data[j], bins, lag
            )
    return MI_lag



def circular_shuffle(x):
    shift = np.random.randint(len(x))
    return np.roll(x, shift)



def shuffled_lagged_mi(x, y, bins, lag, n_shuffle=100):
    null = np.zeros(n_shuffle)
    for k in range(n_shuffle):
        x_s = circular_shuffle(x)
        null[k] = lagged_mi(x_s, y, bins, lag)
    return null






#=============================================================================================================================

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

# 自相关分析部分保持不变
all_selfcorr_curves = [] 

for k, b in enumerate(binsize_range):
    bin_duration_in_frames = int(b * sampling_rate)
    num_bins = total_time // bin_duration_in_frames
    
    selfcorr_matrix = np.zeros((neuron_num, max_lag + 1)) 
    
    for i in range(neuron_num):
        binned_firing_rate = np.zeros(num_bins)
        for bin_idx in range(num_bins):
            start_frame = bin_idx * bin_duration_in_frames
            end_frame = start_frame + bin_duration_in_frames
            spike_count_in_bin = np.sum(spikes[i, start_frame:end_frame])
            binned_firing_rate[bin_idx] = spike_count_in_bin / b
        
        corr_vals = acf(binned_firing_rate, nlags=max_lag, fft=False) 
        selfcorr_matrix[i, :] = corr_vals
    
    all_selfcorr_curves.append(selfcorr_matrix)

# 可视化自相关
lags = np.arange(0, max_lag + 1)
plt.figure(figsize=(12, 6))
for k, b in enumerate(binsize_range):
    plt.plot(lags, all_selfcorr_curves[k][0, :], label=f'bin={b:.2f}s', marker='o', markersize=3)
plt.legend()
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Neuron 0 ACF under different bin sizes')
plt.grid(True, alpha=0.3)
plt.show()

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


#==========普通MI====================
MI, H = compute_mi_matrix(S, num_bins=8)
NMI,DMI = compute_nmi_dmi(MI, H)
print("max |I(X;X) - H(X)| =", np.max(np.abs(np.diag(MI) - H)))

#==========lag MI 扫描===============
lags = np.arange(0, 5)
n_lags = len(lags)
bins = make_shared_bins(S, num_bins=8)
effect_num = neuron_num
MI_lag = np.zeros((effect_num, effect_num, n_lags))
p_lag = np.zeros((effect_num, effect_num, n_lags))
for k, lag in enumerate(lags):
    print(f"{k}is done")
    for i in range(effect_num):
        for j in range(effect_num):
            if i == j:
                MI_lag[i, j, k] = 0
                p_lag[i, j, k] = 1
                continue

            mi_obs = lagged_mi(S[i], S[j], bins, lag)
            mi_null = shuffled_lagged_mi(
                S[i], S[j], bins,
                lag=lag,
                n_shuffle=200
            )

            MI_lag[i, j, k] = mi_obs
            p_lag[i, j, k] = np.mean(mi_null >= mi_obs)

MI_max = MI_lag.max(axis=2)
tau_star = lags[MI_lag.argmax(axis=2)]

# i, j = 96, 68
# plt.plot(lags, MI_lag[i, j, :], marker='o')
# plt.axvline(0, ls='--', c='k')
# plt.xlabel("Lag")
# plt.ylabel("MI")
# plt.title(f"Neuron {i} → {j}")
#==========lag MI 热图===============
plt.figure(figsize=(10, 8))
sns.heatmap(MI_max, square=True, cmap='viridis', 
           cbar_kws={'label': 'MI(lag) max'})
plt.title('MI(lag) max ', fontsize=16, fontweight='bold')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
#==========shuffle laged MI 显著性=============
plt.figure(figsize=(10, 8))
mask = MI_max < np.percentile(MI_max, 95)
sns.heatmap(tau_star, mask=mask, cmap='coolwarm', center=0)
plt.title('shifted lags when Mi=max', fontsize=16, fontweight='bold')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
#==========shuffle laged MI 显著性=============
p_lag_max = np.zeros((effect_num, effect_num))
for i in range(effect_num):
    for j in range(effect_num):
        # 找到 MI 最大时的 lag index
        k_max = MI_lag[i, j, :].argmax()
        p_lag_max[i, j] = p_lag[i, j, k_max]

plt.figure(figsize=(10, 8))
sns.heatmap(p_lag_max, square=True, cmap='coolwarm', cbar_kws={'label': 'p(lag) at MI_max'})
plt.title('p-value at MI_max', fontsize=16, fontweight='bold')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.show()


plt.show()


#===========可视化互信息矩阵===============
figures = []
fig1 = plt.figure(figsize=(10, 8))
sns.heatmap(MI, square=True, cmap='viridis', 
           cbar_kws={'label': 'Mutual Information'})
plt.title('Mutual Information Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
figures.append(fig1)

fig2 = plt.figure(figsize=(10, 8))
sns.heatmap(NMI, square=True, cmap='viridis', vmin=0, vmax=1,
           cbar_kws={'label': 'Normalized MI'})
plt.title('NMI Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
figures.append(fig2)

fig3 = plt.figure(figsize=(10, 8))
sns.heatmap(DMI, square=True, cmap='viridis',
           cbar_kws={'label': 'DMI Value'})
plt.title('DMI Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
figures.append(fig3)
plt.show()
