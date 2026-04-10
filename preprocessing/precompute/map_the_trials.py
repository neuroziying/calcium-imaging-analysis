from multiregion_precompute import multiregion_precompute
import matplotlib.pyplot as plt
import numpy as np

datafile_path = "E:/data & works/LNP2/multi-region/d2/session4-7/day3/session7_behavior_brain.xlsx"
fr = 30

data = multiregion_precompute(datafile_path, fr)
print('data precomputing ... done!')

# for trial_idx in range(data.num_trial):

#     trial = data.trials[trial_idx]
#     n_regions = len(data.col_name)

#     # 没有lick就跳过
#     if len(trial.drink) == 0:
#         print(f"Trial {trial_idx} has no drink event")
#         continue

#     # 定义脑区group
#     region_groups = []

#     # 前20个单独
#     for r in range(15):
#         region_groups.append([r])

#     # 第21~n 合并
#     region_groups.append(list(range(15, n_regions)))

#     n_groups = len(region_groups)

#     # 画图
#     fig, axes = plt.subplots(n_groups, 1, figsize=(8, 1*n_groups), sharex=True)

#     if n_groups == 1:
#         axes = [axes]

#     # ===== 主循环 =====
#     for group_idx, region_group in enumerate(region_groups):

#         ax = axes[group_idx]

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

#         ax.plot(avg_trace, linewidth=2)

#         # ===== label =====
#         if len(region_group) == 1:
#             label = data.col_name[region_group[0]].upper()
#         else:
#             label = "CA1 (avg)"   # 你可以自定义名字

#         ax.set_ylabel(label, rotation=0, ha='right', va='center',fontsize = 12)
#         ax.yaxis.set_label_coords(-0.1, 0.5)

#         # 美化
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         # 对齐点
#         ax.axvline(2*fr, linestyle='--')

#         # 去掉中间x轴
#         if group_idx < n_groups - 1:
#             ax.set_xticks([])

#     T = avg_trace.shape[0]
#     axes[-1].set_xticks([0, T])
#     axes[-1].set_xticklabels(['0s', '8s'])
#     axes[-1].set_xlabel("time (s)")

#     plt.suptitle(f"Trial {trial_idx} - drink aligned activity", y=1, x=0.6, fontsize=12, fontweight = 'bold')
#     plt.tight_layout()
#     plt.savefig(f"E:/data & works/LNP2/multi-region/d2/session4-7/day3/s6/drink/trial_{trial_idx}.png", dpi=300)
#     plt.close()


for trial_idx in range(data.num_trial):

    trial = data.trials[trial_idx]
    n_regions = len(data.col_name)

    # 没有lick就跳过
    if len(trial.press) == 0:
        print(f"Trial {trial_idx} has no press event")
        continue

    # 定义脑区group
    region_groups = []

    # 前20个单独
    for r in range(15):
        region_groups.append([r])

    # 第21~n 合并
    region_groups.append(list(range(15, n_regions)))

    n_groups = len(region_groups)

    # 画图
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 1*n_groups), sharex=True)

    if n_groups == 1:
        axes = [axes]

    # ===== 主循环 =====
    for group_idx, region_group in enumerate(region_groups):

        ax = axes[group_idx]

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

        ax.plot(avg_trace, linewidth=2)

        # ===== label =====
        if len(region_group) == 1:
            label = data.col_name[region_group[0]].upper()
        else:
            label = "CA1 (avg)"   # 你可以自定义名字

        ax.set_ylabel(label, rotation=0, ha='right', va='center',fontsize = 12)
        ax.yaxis.set_label_coords(-0.1, 0.5)

        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 对齐点
        ax.axvline(2*fr, linestyle='--')

        # 去掉中间x轴
        if group_idx < n_groups - 1:
            ax.set_xticks([])

    T = avg_trace.shape[0]
    axes[-1].set_xticks([0, T])
    axes[-1].set_xticklabels(['0s', '8s'])
    axes[-1].set_xlabel("time (s)")

    plt.suptitle(f"Trial {trial_idx} - press aligned activity", y=1, x=0.6, fontsize=12, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig(f"E:/data & works/LNP2/multi-region/d2/session4-7/day3/s7/press/trial_{trial_idx}.png", dpi=300)
    plt.close()


"""
for i in range(data.num_trial):  # 对每个trial
    avg_drink = []
    avg_press = []
    for j in range(len(data.col_name)):  # 选取各个脑区
        avg_drink.append(np.average(data.trials[i].drink[j].caltrace[1])) # 求每种行为的平均活动-t时间序列； delay和decide由于时间上零碎，想另外的办法解决
        avg_press.append(np.average(data.trials[i].press[j].caltrace[1]))
    # 要是能找到一种办法更好地体现trial间变化就好了
    # 总之先画出每一个trial内、各个脑区活动的平均图!
    plt.draw(avg_drink)
    plt.draw(avg_press)

"""

       





