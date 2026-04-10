import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from multiregion_precompute import multiregion_precompute

# ===== 参数 =====
datafile_path = "E:/data & works/LNP2/multi-region/d1/session4_behavior_brain.xlsx"
fr = 30
save_dir = "E:/data & works/LNP2/multi-region/d1/s4"
os.makedirs(save_dir, exist_ok=True)

# ===== 读取数据 =====
data = multiregion_precompute(datafile_path, fr)
print('data precomputing ... done!')

# ===== 脑区分组 =====
n_regions = len(data.col_name)

region_groups = []

for r in range(15):
    region_groups.append([r])

region_groups.append(list(range(15, n_regions)))  # CA1

n_groups = len(region_groups)

# =========================================
# ===== PRESS 相关性 =====
# =========================================
corr_matrices_press = []

for trial_idx in range(data.num_trial):

    trial = data.trials[trial_idx]

    if len(trial.press) == 0:
        continue

    group_traces = []

    for region_group in region_groups:

        traces = []

        for bout in trial.press:
            group_signal = np.mean(
                [bout.caltrace[r, :] for r in region_group],
                axis=0
            )
            traces.append(group_signal)

        if len(traces) == 0:
            continue

        traces = np.array(traces)
        avg_trace = traces.mean(axis=0)

        group_traces.append(avg_trace)

    group_traces = np.array(group_traces)

    if group_traces.shape[0] != n_groups:
        continue

    corr = np.corrcoef(group_traces)
    corr_matrices_press.append(corr)

mean_corr_press = np.mean(np.array(corr_matrices_press), axis=0)

# 排序（press）
Z_press = linkage(mean_corr_press, method='average')
order_press = leaves_list(Z_press)

# =========================================
# ===== DRINK 相关性 =====
# =========================================
corr_matrices_drink = []

for trial_idx in range(data.num_trial):

    trial = data.trials[trial_idx]

    if len(trial.drink) == 0:
        continue

    group_traces = []

    for region_group in region_groups:

        traces = []

        for bout in trial.drink:
            group_signal = np.mean(
                [bout.caltrace[r, :] for r in region_group],
                axis=0
            )
            traces.append(group_signal)

        if len(traces) == 0:
            continue

        traces = np.array(traces)
        avg_trace = traces.mean(axis=0)

        group_traces.append(avg_trace)

    group_traces = np.array(group_traces)

    if group_traces.shape[0] != n_groups:
        continue

    corr = np.corrcoef(group_traces)
    corr_matrices_drink.append(corr)

mean_corr_drink = np.mean(np.array(corr_matrices_drink), axis=0)

# 排序（drink）
Z_drink = linkage(mean_corr_drink, method='average')
order_drink = leaves_list(Z_drink)

# =========================================
# ===== label =====
# =========================================
labels = []
for group in region_groups:
    if len(group) == 1:
        labels.append(data.col_name[group[0]].upper())
    else:
        labels.append("CA1")



# =========================================
# ===== 差值 heatmap（PRESS - DRINK）
# =========================================

diff_corr = mean_corr_press - mean_corr_drink

# 👉 用 PRESS 的排序（推荐）
sorted_diff = diff_corr[order_press][:, order_press]
sorted_labels = [labels[i] for i in order_press]

plt.figure(figsize=(8, 7))

sns.heatmap(
    sorted_diff,
    xticklabels=sorted_labels,
    yticklabels=sorted_labels,
    vmin=-0.5, vmax=0.5,   # 比原来缩小范围，更容易看差异
    cmap='coolwarm',
    center=0,              # ⭐ 关键：0为白色中心
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.title("Difference (PRESS - DRINK)", fontsize=14)

plt.tight_layout()

plt.savefig(f"{save_dir}/diff_press_minus_drink.png", dpi=300)
plt.close()

# 👉 用相关性排序

Z = linkage(diff_corr, method='average')
order_diff = leaves_list(Z)
sorted_diff = diff_corr[order_diff][:, order_diff]


plt.figure(figsize=(8, 7))

sns.heatmap(
    sorted_diff,
    xticklabels=sorted_labels,
    yticklabels=sorted_labels,
    vmin=-0.5, vmax=0.5,   # 比原来缩小范围，更容易看差异
    cmap='coolwarm',
    center=0,              # ⭐ 关键：0为白色中心
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.title("Difference (PRESS - DRINK)", fontsize=14)

plt.tight_layout()

plt.savefig(f"{save_dir}/diff_press_minus_drink_rearranged.png", dpi=300)
plt.close()
