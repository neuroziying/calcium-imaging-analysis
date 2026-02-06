import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import pandas as pd
from scipy.ndimage import gaussian_filter1d,gaussian_filter
from matplotlib.patches import Circle
from matplotlib import cm, colors
import os

#------------导入数据：每一个神经元的放电（Calcium Transients）---------------

Calcium_data_raw = scipy.io.loadmat("C:\\Users\\Administrator\\Desktop\\pycodes\\compact_valid_neurons1011A.mat")  
try:
    S_signals = Calcium_data_raw['compact_neurons']['S_signals'][0,0]
except:
    print("can't find data from mat")
    exit()
spikes = np.array(S_signals)
spikes_shuffled = np.roll(spikes, shift=2000, axis=1)
neuron_num = spikes.shape[0]
total_time = spikes.shape[1]
sampling_rate_N = 16.67

#------------导入数据：小鼠运动-时间---------------------------------
behavior = pd.read_csv(r"C:\Users\Administrator\Desktop\DLC_Projects\MouseBehavior-Ziying_Wang-2025-12-12\analysis_results\mouse_center_trajectory.csv")
sampling_rate_B = 30

frame_series = behavior.iloc[:, 0].values
time_series  = behavior.iloc[:, 1].values
x = behavior.iloc[:, 2].values
y = behavior.iloc[:, 3].values
v = behavior.iloc[:, 4].values

# 将行为轴与时间轴对齐
t_beh = np.arange(len(x)) / 30.0            # behavior time (s)
t_neu = np.arange(spikes.shape[1]) / 16.67  # neural time (s)

# 插值
x_neu = np.interp(t_neu, t_beh, x)
y_neu = np.interp(t_neu, t_beh, y)
v_neu = np.interp(t_neu, t_beh, v) # pix/s
speed = v_neu*30/900  # cm/s
speed_smooth = gaussian_filter1d(speed, sigma=0.5)

# 整理数据：（1）提取小鼠运动速度 >0.5 cm/s 的连续时间段
speed_thresh = 0.5  # cm/s
moving = speed_smooth > speed_thresh

changes = np.diff(moving.astype(int))
starts = np.where(changes == 1)[0] + 1   # False → True
ends   = np.where(changes == -1)[0] + 1  # True → False

if moving[0]:
    starts = np.insert(starts, 0, 0)
if moving[-1]:
    ends = np.append(ends, len(moving))
    
# 整理数据：（2）把这些时间段里，小鼠速度至少达到过 1 cm/s 的部分保留，作为有效数据
valid_speed = 1 # cm/s
valid_segments = [
    (s, e) for s, e in zip(starts, ends)
    if np.max(speed_smooth[s:e]) >= valid_speed
]
valid_mask = np.zeros_like(speed_smooth, dtype=bool)
for s, e in valid_segments:
    valid_mask[s:e] = True

# 将空间分割为 2cm*2cm 的bins； 计算小鼠停留在第i个格子里的总时间
n_bins = 15
num_states = n_bins ** 2
dt = 1 / sampling_rate_N  # 神经时间轴
z_bin = np.zeros(num_states)

for i in range(n_bins):
    for j in range(n_bins):
        index = i * n_bins + j

        x_min = np.min(x_neu) + i * (np.max(x_neu)-np.min(x_neu)) / n_bins
        x_max = np.min(x_neu) + (i+1) * (np.max(x_neu)-np.min(x_neu)) / n_bins
        y_min = np.min(y_neu)  + j * (np.max(y_neu)-np.min(y_neu)) / n_bins
        y_max = np.min(y_neu)  + (j+1) * (np.max(y_neu)-np.min(y_neu)) / n_bins

        mask = (
            valid_mask &
            (x_neu >= x_min) & (x_neu < x_max) &
            (y_neu >= y_min) & (y_neu < y_max)
        )

        z_bin[index] = np.sum(mask) * dt
        


# 对于某一个神经元：求在第i个格子里的放电次数；除以第i个格子内停留的总时间；遍历所有格子。循环处理：细胞序号（格子序号）
placefield = np.zeros([neuron_num, num_states])
for n in range(neuron_num):
    for i in range(n_bins):
        for j in range(n_bins):
            index = i * n_bins + j
            x_min = np.min(x_neu) + i * (np.max(x_neu)-np.min(x_neu)) / n_bins
            x_max = np.min(x_neu) + (i+1) * (np.max(x_neu)-np.min(x_neu)) / n_bins
            y_min = np.min(y_neu)  + j * (np.max(y_neu)-np.min(y_neu)) / n_bins
            y_max = np.min(y_neu)  + (j+1) * (np.max(y_neu)-np.min(y_neu)) / n_bins

            mask = (
                valid_mask &
                (x_neu >= x_min) & (x_neu < x_max) &
                (y_neu >= y_min) & (y_neu < y_max)
            )

            spike_count = np.sum(spikes[n, mask])
            # print(f"total spike count for {n} is {spike_count}")
            if z_bin[index] > 0:
                placefield[n, index] = spike_count / z_bin[index]
            else:
                placefield[n, index] = np.nan
                
# 使用高斯平滑滤波器（δ=3.5）减少锯齿； 除以最大值，归一化。
placefield_smooth = np.zeros_like(placefield)
placefield_normalized = np.zeros_like(placefield)
placefield_width = np.zeros(neuron_num)

for n in range(neuron_num):    
    pf_2d = placefield[n].reshape(n_bins, n_bins)

    pf_filled = np.nan_to_num(pf_2d, nan=0.0)
    pf_smooth = gaussian_filter(pf_filled, sigma=0.5)

    pf_flat = pf_smooth.flatten()

    placefield_smooth[n, :] = pf_flat
    if np.nanmax(pf_flat) > 0:
        placefield_normalized[n, :] = pf_flat / np.nanmax(pf_flat)
    else:
        placefield_normalized[n, :] = np.nan

    # 将 place field 的宽度定义为“取值 >=0.5 的bin 的个数”
    placefield_width[n] = np.sum(placefield_normalized[n] > 0.5)

is_valid_pf = np.zeros(neuron_num, dtype=bool)

for n in range(neuron_num):
    pf = placefield_normalized[n]
    if np.any(~np.isnan(pf)) and np.nansum(pf > 0.5) >= 3:
        is_valid_pf[n] = True

print(f"Valid place cells: {np.sum(is_valid_pf)} / {neuron_num}")


#============================================================================
# statistical analysis
# (1) 计算钙冲动与小鼠位置间的互信息; 小鼠位置划为2cm一格的bin
print(f"begin calculating SI")
SI = np.zeros(neuron_num)

for n in np.where(is_valid_pf)[0]:
    overall_mean_firing_rate = np.sum(spikes[n])/(total_time / sampling_rate_N)
    if overall_mean_firing_rate == 0:
        continue
    for i in range(n_bins):
        for j in range(n_bins):
            index = j * n_bins + i
            x_min = np.min(x_neu) + i * (np.max(x_neu)-np.min(x_neu)) / n_bins
            x_max = np.min(x_neu) + (i+1) * (np.max(x_neu)-np.min(x_neu)) / n_bins
            y_min = np.min(y_neu)  + j * (np.max(y_neu)-np.min(y_neu)) / n_bins
            y_max = np.min(y_neu)  + (j+1) * (np.max(y_neu)-np.min(y_neu)) / n_bins

            mask = (
                valid_mask &
                (x_neu >= x_min) & (x_neu < x_max) &
                (y_neu >= y_min) & (y_neu < y_max)
            )
            probability_density_at_x = z_bin[index] / np.nansum(z_bin)
            spike_count = np.sum(spikes[n, mask])
            if z_bin[index] > 0:
                mean_firing_rate_at_x = spike_count / z_bin[index]
            else:
                mean_firing_rate_at_x = np.nan
            if z_bin[index] > 0 and mean_firing_rate_at_x > 0:
                SI[n] += probability_density_at_x*(mean_firing_rate_at_x/overall_mean_firing_rate)*np.log2(mean_firing_rate_at_x/overall_mean_firing_rate)
            else:
                SI[n] += 0
print(f"begin calculating SI_shuffled")

# (2) 对钙冲动的时间序列做 10,000 shuffles；计算每次shuffle的互信息

times = 5000
SI_shuffled = np.zeros([neuron_num,times])

for n in np.where(is_valid_pf)[0]:
    omr = np.sum(spikes[n])/(total_time / sampling_rate_N)
    if omr == 0:
        continue
    for t in range(times):
        if t%100==0:
            print(f"round{t}")
        shift = np.random.randint(total_time//10, total_time-total_time//10)
        spikes_shuffled = np.roll(spikes[n], shift)
        for i in range(n_bins):
            for j in range(n_bins):
                index = j * n_bins + i
                x_min = np.min(x_neu) + i * (np.max(x_neu)-np.min(x_neu)) / n_bins
                x_max = np.min(x_neu) + (i+1) * (np.max(x_neu)-np.min(x_neu)) / n_bins
                y_min = np.min(y_neu)  + j * (np.max(y_neu)-np.min(y_neu)) / n_bins
                y_max = np.min(y_neu)  + (j+1) * (np.max(y_neu)-np.min(y_neu)) / n_bins

                mask = (
                    valid_mask &
                    (x_neu >= x_min) & (x_neu < x_max) &
                    (y_neu >= y_min) & (y_neu < y_max)
                )
                pdx = z_bin[index] / np.nansum(z_bin)
                spike_count = np.sum(spikes_shuffled[mask])
                if z_bin[index] > 0:
                    mfx = spike_count / z_bin[index]
                else:
                    mfx = np.nan
                if z_bin[index] > 0 and mfx > 0:
                    SI_shuffled[n,t] += pdx*(mfx/omr)*np.log2(mfx/omr)
                else:
                    SI_shuffled[n,t] += 0

# (3) p-value    p<0.05 --> significant place field
is_valid_plc = np.zeros(neuron_num, dtype=bool)

for n in np.where(is_valid_pf)[0]:
    SI[n] = np.nan_to_num(SI[n])
    SI_shuffled[n] = np.nan_to_num(SI_shuffled[n])
    p_value = (np.sum(SI_shuffled[n] >= SI[n]) + 1) / (times + 1)
    if p_value < 0.1:
        is_valid_plc[n] = True

print(f"Valid place cells: {np.sum(is_valid_plc)} / {neuron_num}")


save_dir = r"C:\Users\Administrator\Desktop\pycodes\place_field\placefield_maps"
os.makedirs(save_dir, exist_ok=True)


valid_idx = np.where(is_valid_pf)[0]
SI_valid = SI[valid_idx]
# 每个神经元 shuffled SI 的均值
SI_shuf_mean = np.mean(SI_shuffled[valid_idx], axis=1)
# 95% 分位数
SI_shuf_95 = np.percentile(SI_shuffled[valid_idx], 95, axis=1)

plt.figure(figsize=(6,4))

plt.hist(SI_valid, bins=40, alpha=0.6, label='Observed SI')
plt.hist(SI_shuf_95, bins=40, alpha=0.6, label='Shuffled SI (mean)')

plt.xlabel('Spatial Information (bits)')
plt.ylabel('Neuron count')
plt.legend()
plt.title('SI distribution: observed vs shuffled')

plt.tight_layout()
save_path = os.path.join(save_dir, f"neuron_{n:03d}_placefield.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()  


SI_shuf_std = np.std(SI_shuffled[valid_idx], axis=1)
k = 1   # 1.5 / 2 / 2.5 
is_spatial = SI_valid > (SI_shuf_mean + k * SI_shuf_std)
selected_idx = valid_idx[is_spatial]
print(f"SI > shuffled mean + {k}std: {len(selected_idx)} neurons")


# # 画top10%的SI神经元
# sorted_idx = valid_idx[np.argsort(SI_valid)[::-1]]
# top_frac = 0.10
# n_top = int(len(sorted_idx) * top_frac)
# top10_idx = sorted_idx[:n_top]
# print(f"Top 10% spatially tuned neurons: {n_top}")
# for n in top10_idx:
#     plt.imshow(
#         placefield_normalized[n].reshape(n_bins, n_bins),
#         origin='lower',
#         cmap='hot'
#     )
#     plt.title(f'Neuron {n} (SI={SI[n]:.3f})')
#     plt.colorbar()
#     plt.show()

# for n in range(1, 10):
#     pf_smooth_2d = placefield_smooth[n].reshape(n_bins, n_bins)
#     plt.imshow(pf_smooth_2d, origin='lower', cmap='hot')
#     plt.title(f"Smoothed PF neuron {n}")
#     plt.colorbar()
#     plt.show()

# # 检查时间轴质量
# neural_activity_sum = np.nansum(spikes, axis=0)

# plt.figure()
# plt.plot(t_neu, neural_activity_sum)
# plt.xlabel("Time (s)")
# plt.ylabel("Total neural activity")

# plt.figure()
# plt.plot(t_beh, np.ones_like(t_beh))
# plt.xlabel("Time (s)")
# plt.title("Behavior timeline")
# plt.show()

# active_frames = np.where(neural_activity_sum > 0)[0]

# plt.figure()
# plt.scatter(
#     x_neu[active_frames],
#     y_neu[active_frames],
#     s=3,
#     alpha=0.6
# )
# plt.gca().invert_yaxis()
# plt.title("Positions during neural activity")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis("equal")
# plt.show()

# # 检查插值质量
# plt.figure(figsize=(10,4))

# plt.subplot(1,2,1)
# plt.plot(x, y, '.', markersize=1)
# plt.title('Raw behavior trajectory')
# plt.axis('equal')

# plt.subplot(1,2,2)
# plt.plot(x_neu, y_neu, '.', markersize=1)
# plt.title('Interpolated neural trajectory')
# plt.axis('equal')

# plt.show()

# # 画速度分布
# plt.figure()
# plt.hist(speed_smooth, bins=100)
# plt.axvline(0.5, color='r')
# plt.axvline(1.0, color='k')
# plt.xlabel('Speed (cm/s)')
# plt.ylabel('Counts')
# plt.title('Speed distribution')
# plt.show()

# # 画时间占有情况
# occupancy = np.zeros((n_bins, n_bins))

# for i in range(n_bins):
#     for j in range(n_bins):
#         x_min = np.min(x_neu) + i * (np.max(x_neu)-np.min(x_neu)) / n_bins
#         x_max = np.min(x_neu) + (i+1) * (np.max(x_neu)-np.min(x_neu)) / n_bins
#         y_min = np.min(y_neu)  + j * (np.max(y_neu)-np.min(y_neu)) / n_bins
#         y_max = np.min(y_neu)  + (j+1) * (np.max(y_neu)-np.min(y_neu)) / n_bins

#         mask = (
#             valid_mask &
#             (x_neu >= x_min) & (x_neu < x_max) &
#             (y_neu >= y_min) & (y_neu < y_max)
#         )
#         occupancy[j, i] = np.sum(mask)

# plt.imshow(occupancy, origin='lower', cmap='viridis')
# plt.colorbar(label='Frames')
# plt.title('Occupancy map (valid_mask)')
# plt.show()

# # 画直方图展示分布
# plt.hist(
#     placefield_width,
#     bins=20,
#     color='steelblue',
#     edgecolor='black',
#     alpha=0.8
# )

# plt.xlabel('width')
# plt.ylabel('number of neurons')
# plt.title('distribution of place field widths')
# plt.tight_layout()
# plt.show()



# save_dir = r"C:\Users\Administrator\Desktop\pycodes\place_field\placefield_maps"
# os.makedirs(save_dir, exist_ok=True)

# for n in np.where(is_valid_pf)[0]:
#     plt.figure(figsize=(4, 4))
#     plt.imshow(
#         placefield_normalized[n].reshape(n_bins, n_bins),
#         origin='lower',
#         cmap='hot'
#     )
#     plt.colorbar(label='Normalized firing rate')
#     plt.title(f'Neuron {n}')
#     plt.xlabel('X bin')
#     plt.ylabel('Y bin')
#     save_path = os.path.join(save_dir, f"neuron_{n:03d}_placefield.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()  

# # 将不同细胞的place field用质心+半径的方式展示在一张图中。
# xs = np.arange(n_bins)
# ys = np.arange(n_bins)
# X, Y = np.meshgrid(xs, ys)
# centers = np.zeros((neuron_num, 2))

# for n in range(neuron_num):
#     pf = placefield_normalized[n].reshape(n_bins, n_bins)

#     if np.all(np.isnan(pf)) or np.max(pf) == 0:
#         centers[n] = np.nan
#         continue

#     weight = pf / np.nansum(pf)
#     cx = np.nansum(weight * X)
#     cy = np.nansum(weight * Y)

#     centers[n] = [cx, cy]

# radii = placefield_width
# plt.figure(figsize=(6, 6))
# ax = plt.gca()

# norm = colors.Normalize(
#     vmin=np.nanmin(radii),
#     vmax=np.nanmax(radii)
# )
# cmap = cm.viridis  

# for n in range(neuron_num):
#     if np.any(np.isnan(centers[n])):
#         continue

#     color = cmap(norm(radii[n]))

#     circle = Circle(
#         centers[n],
#         radius=radii[n],
#         facecolor=color,
#         edgecolor='none',   # 去掉边框
#         alpha=0.4           # 40% 透明度
#     )
#     ax.add_patch(circle)

# ax.set_xlim(0, n_bins)
# ax.set_ylim(0, n_bins)
# ax.set_aspect('equal')
# ax.set_xlabel('X bin')
# ax.set_ylabel('Y bin')
# ax.set_title('Place field centers (color = size)')

# sm = cm.ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])  
# plt.colorbar(sm, ax=ax, label='Place field radius (bins)')

# plt.show()