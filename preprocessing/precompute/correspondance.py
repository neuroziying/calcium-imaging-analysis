from multiregion_precompute import multiregion_precompute
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
import os

# ===== 参数 =====
datafile_path = "E:/data & works/LNP2/multi-region/d2/session4-7/day2/session4_behavior_brain.xlsx"
fr = 30
save_dir = "E:/data & works/LNP2/multi-region/d2/session4-7/day2/s4"
os.makedirs(save_dir, exist_ok=True)

# # ===== 读取数据 =====
# data = multiregion_precompute(datafile_path, fr)
# print('data precomputing ... done!')

# # ===== 脑区分组 =====
# n_regions = len(data.col_name)

# region_groups = []

# # 前20个单独
# for r in range(15):
#     region_groups.append([r])

# # 后面合并（CA1）
# region_groups.append(list(range(15, n_regions)))

# n_groups = len(region_groups)

# # ===== 收集所有trial的相关性矩阵 =====
# corr_matrices = []

# for trial_idx in range(data.num_trial):

#     trial = data.trials[trial_idx]

#     if len(trial.drink) == 0:
#         continue

#     group_traces = []

#     # ===== 每个脑区group计算avg trace =====
#     for region_group in region_groups:

#         traces = []

#         for bout in trial.drink:
#             group_signal = np.mean(
#                 [bout.caltrace[r, :] for r in region_group],
#                 axis=0
#             )
#             traces.append(group_signal)

#         if len(traces) == 0:
#             continue

#         traces = np.array(traces)
#         avg_trace = traces.mean(axis=0)

#         group_traces.append(avg_trace)

#     # 转为矩阵 (n_groups, T)
#     group_traces = np.array(group_traces)

#     if group_traces.shape[0] != n_groups:
#         continue  # 防止异常

#     # ===== 计算相关性 =====
#     corr = np.corrcoef(group_traces)

#     corr_matrices.append(corr)

# # ===== 平均 =====
# corr_matrices = np.array(corr_matrices)
# mean_corr = np.mean(corr_matrices, axis=0)

# # ===== 层次聚类排序 =====
# Z = linkage(mean_corr, method='average')
# order = leaves_list(Z)

# sorted_corr = mean_corr[order][:, order]

# # ===== label =====
# labels = []
# for group in region_groups:
#     if len(group) == 1:
#         labels.append(data.col_name[group[0]].upper())
#     else:
#         labels.append("CA1")

# sorted_labels = [labels[i] for i in order]

# # ===== 画 heatmap =====
# plt.figure(figsize=(8, 7))

# sns.heatmap(
#     sorted_corr,
#     xticklabels=sorted_labels,
#     yticklabels=sorted_labels,
#     vmin=-1, vmax=1,
#     cmap='coolwarm',
#     square=True,
#     cbar_kws={"shrink": 0.8}
# )

# plt.xticks(rotation=90)
# plt.yticks(rotation=0)

# plt.title("Mean Brain Region Correlation (Drink Trials)", fontsize=14)

# plt.tight_layout()

# # ===== 保存 =====
# plt.savefig(f"{save_dir}/correlation_heatmap_sorted.png", dpi=300)
# plt.close()


# ===== 读取数据 =====
data = multiregion_precompute(datafile_path, fr)
print('data precomputing ... done!')

# ===== 脑区分组 =====
n_regions = len(data.col_name)

region_groups = []

# 前20个单独
for r in range(15):
    region_groups.append([r])

# 后面合并（CA1）
region_groups.append(list(range(15, n_regions)))

n_groups = len(region_groups)

# ===== 收集所有trial的相关性矩阵 =====
corr_matrices = []

for trial_idx in range(data.num_trial):

    trial = data.trials[trial_idx]

    if len(trial.press) == 0:
        continue

    group_traces = []

    # ===== 每个脑区group计算avg trace =====
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

    # 转为矩阵 (n_groups, T)
    group_traces = np.array(group_traces)

    if group_traces.shape[0] != n_groups:
        continue  # 防止异常

    # ===== 计算相关性 =====
    corr = np.corrcoef(group_traces)

    corr_matrices.append(corr)

# ===== 平均 =====
corr_matrices = np.array(corr_matrices)
mean_corr = np.mean(corr_matrices, axis=0)

# ===== 层次聚类排序 =====
Z = linkage(mean_corr, method='average')
order = leaves_list(Z)

sorted_corr = mean_corr[order][:, order]

# ===== label =====
labels = []
for group in region_groups:
    if len(group) == 1:
        labels.append(data.col_name[group[0]].upper())
    else:
        labels.append("CA1")

sorted_labels = [labels[i] for i in order]

# ===== 画 heatmap =====
plt.figure(figsize=(8, 7))

sns.heatmap(
    sorted_corr,
    xticklabels=sorted_labels,
    yticklabels=sorted_labels,
    vmin=-1, vmax=1,
    cmap='coolwarm',
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.title("Mean Brain Region Correlation (Press Trials)", fontsize=14)

plt.tight_layout()

# ===== 保存 =====
plt.savefig(f"{save_dir}/correlation_heatmap_sorted_press.png", dpi=300)
plt.close()