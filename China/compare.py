# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")

from lightgbm import LGBMRegressor
from lightgbm import early_stopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

# ==============================
# 画图中文避免乱码
# ==============================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================
# ✅ 固定时间分区（你新要求的）
# 数据：2021-09-01 00:00:00 ~ 2022-08-31 00:00:00
# 训练=前8个月；验证=中2个月；测试=最后2个月
# ==============================
TRAIN_START = pd.Timestamp("2021-09-01 00:00:00")
TRAIN_END   = pd.Timestamp("2022-04-30 23:59:59")

VAL_START   = pd.Timestamp("2022-05-01 00:00:00")
VAL_END     = pd.Timestamp("2022-06-30 23:59:59")

TEST_START  = pd.Timestamp("2022-07-01 00:00:00")
TEST_END    = pd.Timestamp("2022-08-31 23:59:59")

# 只用这一年
DATA_START  = TRAIN_START
DATA_END    = TEST_END


# ==============================
# 通用图片保存函数：PNG + EPS
# ==============================
def save_png_and_eps(filename, dpi=300, fig=None):
   if fig is None:
       fig_to_save = plt.gcf()
   else:
       fig_to_save = fig

   root, ext = os.path.splitext(filename)
   if ext == "":
       png_name = root + ".png"
   else:
       png_name = filename
       root, _ = os.path.splitext(png_name)

   eps_name = root + ".eps"

   fig_to_save.savefig(png_name, dpi=dpi)
   fig_to_save.savefig(eps_name, format="eps", dpi=dpi)
   print(f"  → 已保存图片：{png_name} 和 {eps_name}")
   return png_name, eps_name


# ==============================
# 工具函数
# ==============================
def find_wind_col(df):
   for c in df.columns:
       if "WindSpeed" in str(c):
           return c
   raise ValueError("找不到 wind80m 列，请检查表头（列名中要包含 'wind80m'）")


def calc_stats(fcst, obs):
   fcst = np.asarray(fcst, dtype=float)
   obs = np.asarray(obs, dtype=float)
   diff = fcst - obs
   mae = np.abs(diff).mean()
   mse = (diff ** 2).mean()
   rmse = np.sqrt(mse)
   return mae, mse, rmse


def calc_corr(fcst, obs):
   fcst = np.asarray(fcst, dtype=float)
   obs = np.asarray(obs, dtype=float)
   if len(fcst) < 2:
       return np.nan
   if np.std(fcst) == 0 or np.std(obs) == 0:
       return np.nan
   return float(np.corrcoef(fcst, obs)[0, 1])


def windspeed_to_power(u,
                      B=-0.0976,
                      T=1.0032,
                      b=0.425,
                      xmid=11.3921,
                      s=0.307):
   u = np.asarray(u, dtype=float)
   return B + (T - B) / (1.0 + 10.0 ** (b * (xmid - u))) ** s


def kalman_regression(yt, Zt, q=1e-3, r=1.0):
   yt = np.asarray(yt, dtype=float)
   Zt = np.asarray(Zt, dtype=float)
   n, p = Zt.shape

   beta = np.zeros(p)
   P = np.eye(p) * 1000.0
   Q = np.eye(p) * q
   R = float(r)

   beta_hist = np.zeros((n, p))
   bias_pred = np.zeros(n)
   bias_filt = np.zeros(n)

   for t in range(n):
       z = Zt[t, :].reshape(-1, 1)

       beta_pred = beta
       P_pred = P + Q

       y_pred = (z.T @ beta_pred.reshape(-1, 1)).item()
       bias_pred[t] = y_pred

       S = (z.T @ P_pred @ z).item() + R
       K = (P_pred @ z / S).flatten()

       beta = beta_pred + K * (yt[t] - y_pred)
       P = P_pred - np.outer(K, z.flatten()) @ P_pred

       beta_hist[t, :] = beta
       bias_filt[t] = (z.T @ beta.reshape(-1, 1)).item()

   return beta_hist, bias_pred, bias_filt


def compute_ramp_flag(series, ramp_th):
   series = np.asarray(series, dtype=float)
   P = windspeed_to_power(series)
   diffP = np.diff(P)
   flag = np.zeros_like(series, dtype=int)
   if len(diffP) > 0:
       up_mask = diffP >= ramp_th
       down_mask = diffP <= -ramp_th
       flag[1:][up_mask] = 1
       flag[1:][down_mask] = 2
   return flag


def compute_ramp_label_3class(series, ramp_th):
   return compute_ramp_flag(series, ramp_th)


def compute_multi_metrics(y_true, y_pred):
   y_true = np.asarray(y_true, dtype=int)
   y_pred = np.asarray(y_pred, dtype=int)

   acc = accuracy_score(y_true, y_pred)

   p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
       y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
   )

   p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
       y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
   )
   (p0, p1, p2) = p_cls
   (r0, r1, r2) = r_cls
   (f10, f11, f12) = f1_cls

   return {
       "acc": acc,
       "macro_p": p_macro, "macro_r": r_macro, "macro_f1": f1_macro,
       "p0": p0, "r0": r0, "f10": f10,
       "p1": p1, "r1": r1, "f11": f11,
       "p2": p2, "r2": r2, "f12": f12,
   }


def plot_ramp_confusion_matrix(y_true, y_pred, title, save_name):
   labels = [0, 1, 2]
   cm = confusion_matrix(y_true, y_pred, labels=labels)

   plt.figure(figsize=(6, 5))
   ax = plt.gca()

   im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
   plt.colorbar(im, ax=ax)

   class_names = ["不突变", "突增", "突减"]
   tick_marks = np.arange(len(class_names))

   ax.set_xticks(tick_marks)
   ax.set_yticks(tick_marks)
   ax.set_xticklabels(class_names)
   ax.set_yticklabels(class_names)

   ax.set_xlabel("预测类别")
   ax.set_ylabel("真实类别")
   ax.set_title(title)

   thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
   for i in range(cm.shape[0]):
       for j in range(cm.shape[1]):
           ax.text(
               j, i, cm[i, j],
               ha="center", va="center",
               color="white" if cm[i, j] > thresh else "black",
               fontsize=12
           )

   col_sum_up = cm[:, 1].sum()
   precision_up = cm[1, 1] / col_sum_up if col_sum_up > 0 else np.nan

   col_sum_down = cm[:, 2].sum()
   precision_down = cm[2, 2] / col_sum_down if col_sum_down > 0 else np.nan

   text_str = (
       f"突增准确率 = {precision_up:.3f}\n"
       f"突减准确率 = {precision_down:.3f}"
   )

   ax.text(
       1.05, 0.25, text_str,
       transform=ax.transAxes,
       fontsize=11,
       ha="left", va="center",
       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6)
   )

   plt.tight_layout()
   save_png_and_eps(save_name, dpi=300)

   print(f"    突增准确率 = {precision_up:.3f}")
   print(f"    突减准确率 = {precision_down:.3f}")

   plt.close()

   return cm, precision_up, precision_down


def EMA_bias_prediction(fc, obs, alpha=0.1):
   fc = np.asarray(fc, dtype=float)
   obs = np.asarray(obs, dtype=float)
   n = len(fc)

   b = 0.0
   bias_pred = np.zeros(n)

   for t in range(n):
       bias_pred[t] = b
       err_t = fc[t] - obs[t]
       b = (1 - alpha) * b + alpha * err_t

   return bias_pred


def build_sequences(feature_array, target_array, window_size):
   X, y = [], []
   N = len(target_array)
   for i in range(window_size - 1, N):
       X.append(feature_array[i - window_size + 1: i + 1, :])
       y.append(target_array[i])
   return np.array(X), np.array(y)


def build_tcn_model(window_size, n_features=1, filters=32, kernel_size=3,
                   dilations=(1, 2, 4, 8)):
   model = Sequential()
   for i, d in enumerate(dilations):
       model.add(
           Conv1D(
               filters=filters,
               kernel_size=kernel_size,
               dilation_rate=d,
               padding="causal",
               activation="relu",
               input_shape=(window_size, n_features) if i == 0 else None
           )
       )
       model.add(BatchNormalization())
   model.add(GlobalAveragePooling1D())
   model.add(Dense(32, activation="relu"))
   model.add(Dense(1))
   model.compile(optimizer="adam", loss="mse")
   return model


def compute_ramp_accuracy(fc_series, obs_series, ramp_th):
   fc_series = np.asarray(fc_series, dtype=float)
   obs_series = np.asarray(obs_series, dtype=float)
   ramp_obs = compute_ramp_flag(obs_series, ramp_th)
   ramp_fc = compute_ramp_flag(fc_series, ramp_th)
   if len(ramp_obs) == 0:
       return np.nan
   return float((ramp_obs == ramp_fc).mean())


def compute_ramp_skill_lstm(fc_series, obs_series, time_series,
                          test_start_time, window_size, ramp_th):
   """
   ✅ 固定只用【训练期（前8个月）】训练 ramp-skill（不看 验证/测试）
   """
   fc_series = np.asarray(fc_series, dtype=float)
   obs_series = np.asarray(obs_series, dtype=float)
   time_series = pd.to_datetime(time_series)
   n = len(fc_series)
   if n < window_size + 30:
       return 1.0

   # 训练结束固定为验证开始时间（VAL_START），即只用训练期
   train_end_time = VAL_START
   train_end_time = min(train_end_time, test_start_time)

   train_mask_all_ts = time_series < train_end_time
   if train_mask_all_ts.sum() < 50:
       return 1.0

   # 标签（3类）
   P_obs = windspeed_to_power(obs_series)
   obs_diff = np.diff(P_obs)
   ramp_labels = np.zeros_like(obs_series, dtype=int)
   if len(obs_diff) > 0:
       up_mask = obs_diff >= ramp_th
       down_mask = obs_diff <= -ramp_th
       ramp_labels[1:][up_mask] = 1
       ramp_labels[1:][down_mask] = 2

   fc_diff = np.diff(fc_series)
   fc_diff = np.concatenate([[0.0], fc_diff])
   fc_diff_sign = np.sign(fc_diff)

   P_fc = windspeed_to_power(fc_series)
   diffP_fc = np.diff(P_fc)
   diffP_fc = np.concatenate([[0.0], diffP_fc])
   diffP_fc_sign = np.sign(diffP_fc)

   features = np.column_stack([
       fc_series,
       fc_diff,
       fc_diff_sign,
       P_fc,
       diffP_fc,
       diffP_fc_sign
   ])

   scalerX = StandardScaler()
   scalerX.fit(features[train_mask_all_ts])
   features_scaled = scalerX.transform(features)

   X_seq, y_seq = build_sequences(features_scaled, ramp_labels, window_size)
   y_seq_cat = to_categorical(y_seq, num_classes=3)

   time_seq_local = time_series[window_size - 1:]
   train_mask_local = time_seq_local < train_end_time
   train_idx_local = np.where(train_mask_local)[0]

   if len(train_idx_local) < 50:
       return 1.0

   split = int(len(train_idx_local) * 0.8)
   idx_train = train_idx_local[:split]
   idx_val = train_idx_local[split:]
   if len(idx_val) == 0:
       return 1.0

   model = Sequential()
   model.add(LSTM(8, input_shape=(window_size, features_scaled.shape[1])))
   model.add(Dense(3, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   es_local = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

   model.fit(
       X_seq[idx_train], y_seq_cat[idx_train],
       validation_data=(X_seq[idx_val], y_seq_cat[idx_val]),
       epochs=30,
       batch_size=64,
       callbacks=[es_local],
       verbose=0
   )

   _, acc = model.evaluate(X_seq[idx_val], y_seq_cat[idx_val], verbose=0)
   return float(acc)


def _enforce_min_weights(w, indices, min_w):
   w = np.asarray(w, dtype=float)
   if indices is None or len(indices) == 0:
       return w

   indices = [idx for idx in indices if 0 <= idx < len(w)]
   if not indices:
       return w

   incr = 0.0
   for idx in indices:
       if w[idx] < min_w:
           incr += (min_w - w[idx])
           w[idx] = min_w

   if incr <= 0:
       s = w.sum()
       if s > 0:
           w /= s
       return w

   other_indices = [i for i in range(len(w)) if i not in indices]
   other_sum = w[other_indices].sum()

   if other_sum > 0 and other_sum > incr:
       factor = (other_sum - incr) / other_sum
       for i in other_indices:
           w[i] *= factor
   else:
       s = w.sum()
       if s > 0:
           w /= s

   w = np.maximum(w, 0.0)
   s = w.sum()
   if s > 0:
       w /= s
   return w


def kalman_dynamic_fusion(
   fc_list,
   obs,
   q_w=1e-4,
   r_w=1.0,
   ramp_flags=None,
   ramp_up_indices=None,
   ramp_down_indices=None,
   calm_pref_indices=None,
   ramp_up_boost_factor=2.5,
   ramp_down_boost_factor=2.5,
   calm_boost_factor=2.0,
   min_ramp_up_weight_indices=None,
   min_ramp_down_weight_indices=None,
   min_weight_up=0.3,
   min_weight_down=0.3,
):
   obs = np.asarray(obs, dtype=float)
   fc_arrs = [np.asarray(fc, dtype=float) for fc in fc_list]
   n = len(obs)
   M = len(fc_arrs)

   w = np.ones(M) / M
   P = np.eye(M) * 100.0
   Q = np.eye(M) * q_w
   R = float(r_w)
   I = np.eye(M)

   w_hist = np.zeros((n, M))
   fc_fuse_prior = np.zeros(n)
   fc_fuse_post = np.zeros(n)

   has_ramp_logic = ramp_flags is not None
   if has_ramp_logic:
       ramp_flags = np.asarray(ramp_flags, dtype=int)
       if len(ramp_flags) != n:
           raise ValueError("ramp_flags 长度必须与 obs / fc_list 长度一致")

   ramp_up_indices = set(ramp_up_indices or [])
   ramp_down_indices = set(ramp_down_indices or [])
   calm_pref_indices = set(calm_pref_indices or [])

   for t in range(n):
       H = np.array([fc_arrs[m][t] for m in range(M)]).reshape(1, -1)

       w_pred = w
       P_pred = P + Q

       if has_ramp_logic:
           if ramp_flags[t] == 1:
               for idx in ramp_up_indices:
                   if 0 <= idx < M:
                       w_pred[idx] *= ramp_up_boost_factor
           elif ramp_flags[t] == 2:
               for idx in ramp_down_indices:
                   if 0 <= idx < M:
                       w_pred[idx] *= ramp_down_boost_factor
           else:
               for idx in calm_pref_indices:
                   if 0 <= idx < M:
                       w_pred[idx] *= calm_boost_factor

       w_pred = np.maximum(w_pred, 0.0)
       s = w_pred.sum()
       if s > 0:
           w_pred /= s
       else:
           w_pred = np.ones(M) / M

       y_prior = float(H @ w_pred.reshape(-1, 1))
       fc_fuse_prior[t] = y_prior

       S = float(H @ P_pred @ H.T + R)
       K = (P_pred @ H.T) / S

       w = w_pred + (K.flatten() * (obs[t] - y_prior))
       P = (I - K @ H) @ P_pred

       w = np.maximum(w, 0.0)
       s = w.sum()
       if s > 0:
           w /= s
       else:
           w = np.ones(M) / M

       if has_ramp_logic:
           if ramp_flags[t] == 1 and min_ramp_up_weight_indices:
               w = _enforce_min_weights(w, min_ramp_up_weight_indices, min_weight_up)
           elif ramp_flags[t] == 2 and min_ramp_down_weight_indices:
               w = _enforce_min_weights(w, min_ramp_down_weight_indices, min_weight_down)
           elif ramp_flags[t] == 0 and calm_pref_indices:
               w = _enforce_min_weights(w, calm_pref_indices, 0.7)
       w_hist[t, :] = w
       fc_fuse_post[t] = float(H @ w.reshape(-1, 1))

   return w_hist, fc_fuse_prior, fc_fuse_post


def plot_obs_and_fuse_with_ramps(time_test, obs_test, fc_test, fc_RAW_test,
                                title, save_name, ramp_th):
   time_test = np.asarray(time_test)
   obs_test = np.asarray(obs_test, dtype=float)
   fc_test = np.asarray(fc_test, dtype=float)
   fc_RAW_test = np.asarray(fc_RAW_test, dtype=float)

   plt.figure(figsize=(11, 5))
   ax = plt.gca()
   ax.plot(time_test, obs_test, linewidth=1.5, label="实测 Obs")
   ax.plot(time_test, fc_test, linewidth=1.5, label="融合/方案预测")

   ramp_obs = compute_ramp_flag(obs_test, ramp_th)
   obs_ramp = obs_test.copy()
   obs_ramp[ramp_obs == 0] = np.nan
   ax.plot(time_test, obs_ramp, linewidth=2.2, label="实测突变段")

   ramp_fc = compute_ramp_flag(fc_test, ramp_th)
   fc_ramp = fc_test.copy()
   fc_ramp[ramp_fc == 0] = np.nan
   ax.plot(time_test, fc_ramp, linewidth=2.2, label="融合/方案突变段")

   ramp_fc_RAW = compute_ramp_flag(fc_RAW_test, ramp_th)
   fc_RAW_ramp = fc_RAW_test.copy()
   fc_RAW_ramp[ramp_fc_RAW == 0] = np.nan
   ax.plot(time_test, fc_RAW_ramp, linewidth=2.0, label="原始预报突变段")

   ax.set_title(title)
   ax.set_xlabel("Time")
   ax.set_ylabel("Wind Speed at 80m (m/s)")
   ax.grid(True, alpha=0.3)
   ax.legend()
   plt.tight_layout()
   save_png_and_eps(save_name, dpi=300)


def print_fusion_weight_by_ramp(w_hist, ramp_flags, model_names, lead_hour, fusion_name):
   ramp_flags = np.asarray(ramp_flags, dtype=int)
   for flag, label in [(0, "不突变"), (1, "突增"), (2, "突减")]:
       mask = ramp_flags == flag
       if not np.any(mask):
           print(f"  Lead={lead_hour}h {fusion_name} | {label} 样本数为 0，无法统计权重。")
           continue
       w_mean = w_hist[mask].mean(axis=0)
       print(f"  Lead={lead_hour}h {fusion_name} | {label} 事件平均权重：")
       for name, w in zip(model_names, w_mean):
           print(f"    {name:24s}: {w:.3f}")


# ==============================
# 配置：改成预报-实况一一对应
# ==============================
pairs = {
   "006": ("wind_006.xlsx", "18点实测风速.xlsx"),
   "012": ("wind_012.xlsx", "0点实测风速.xlsx"),
   "018": ("wind_018.xlsx", "6点实测风速.xlsx"),
   "024": ("wind_024.xlsx", "12点实测风速.xlsx"),
   "030": ("wind_030.xlsx", "18点实测风速.xlsx"),
}

WINDOW_SIZE = 24

KF_Q = 1e-3
KF_R = 1.0

ALPHA_EMA = 0.1

FUSION_QW = 1e-4
FUSION_RW = 1.0

LSTM_UNITS = 32
TCN_FILTERS = 32
TCN_KERNEL = 3
EPOCHS = 50
BATCH_SIZE = 32

RAMP_TH = 0.2

RAMP_UP_WEIGHT = 3.5
RAMP_DOWN_WEIGHT = 3.5

CALM_BOOST_FACTOR = 4.0
RAMP_UP_BOOST_FACTOR = 2.0
RAMP_DOWN_BOOST_FACTOR = 2.0

FUSE_TOPK = 8

summary_rows = []
ramp_metrics_summary_rows = []


# ==============================
# 主循环：遍历各 lead / 文件对
# ==============================
for lead_code, (fcst_file, obs_file) in pairs.items():
   lead_hour = int(lead_code)   # 保留 lead 小时数，用于打印/文件命名

   print(f"\n====================")
   print(f"  Lead = {lead_hour} h (代码 {lead_code})")
   print(f"  预报文件: {fcst_file}")
   print(f"  实况文件: {obs_file}")
   print(f"====================")

   # ========= 读取实况 =========
   obs_df = pd.read_excel(obs_file)
   if "Time" not in obs_df.columns:
       raise ValueError(f"实况文件 {obs_file} 中必须包含 'Time' 列，请检查。")
   obs_df["Time"] = pd.to_datetime(obs_df["Time"])
   obs_col = find_wind_col(obs_df)
   obs_df = obs_df[["Time", obs_col]].rename(columns={obs_col: "Obs"})

   # ✅ 强制过滤到 2021-09 ~ 2022-08
   obs_df = obs_df[(obs_df["Time"] >= DATA_START) & (obs_df["Time"] <= DATA_END)].copy()
   obs_df = obs_df.sort_values("Time").reset_index(drop=True)

   # ========= 读取预报 =========
   fc_df = pd.read_excel(fcst_file)
   if "Time" not in fc_df.columns:
       raise ValueError(f"{fcst_file} 中必须包含 'Time' 列，请检查。")

   fc_df["Time"] = pd.to_datetime(fc_df["Time"])
   fc_col = find_wind_col


   fc_col = find_wind_col(fc_df)
   fc_df = fc_df[["Time", fc_col]].rename(columns={fc_col: "Fcst"})

   # 过滤 2021-09 到 2022-08
   fc_df = fc_df[(fc_df["Time"] >= DATA_START) & (fc_df["Time"] <= DATA_END)].copy()
   fc_df = fc_df.sort_values("Time").reset_index(drop=True)

   # ========= 合并预报和实况 =========
   df = pd.merge(fc_df, obs_df, on="Time", how="inner").dropna().sort_values("Time")
   df = df.reset_index(drop=True)

   if len(df) <= 2 * WINDOW_SIZE + 30:
       print(f"  数据量 {len(df)} 太少，跳过该 Lead。")
       continue

   print(f"  总数据量：{len(df)}  时间：{df['Time'].min()} ~ {df['Time'].max()}")

   # ========= 构造特征（完全保留你的结构） =========
   tmp = df.copy()
   tmp["hour"] = tmp["Time"].dt.hour
   tmp["month"] = tmp["Time"].dt.month
   tmp["dow"] = tmp["Time"].dt.dayofweek

   for L in [1, 2, 3, 6, 12, 24]:
       tmp[f"Fcst_lag{L}"] = tmp["Fcst"].shift(L)

   for W in [3, 6, 12, 24]:
       tmp[f"Fcst_rollmean{W}"] = tmp["Fcst"].rolling(W, min_periods=1).mean()
       tmp[f"Fcst_rollstd{W}"] = tmp["Fcst"].rolling(W, min_periods=1).std()

   tmp["Fcst_diff1"] = tmp["Fcst"].diff()
   tmp["Fcst_diff_abs1"] = tmp["Fcst_diff1"].abs()
   tmp["Fcst_diff_sign"] = np.sign(tmp["Fcst_diff1"].fillna(0.0))

   tmp["Fcst_power"] = windspeed_to_power(tmp["Fcst"])
   tmp["Fcst_power_diff1"] = tmp["Fcst_power"].diff()
   tmp["Fcst_power_diff_abs1"] = tmp["Fcst_power_diff1"].abs()
   tmp["Fcst_power_diff_sign"] = np.sign(tmp["Fcst_power_diff1"].fillna(0.0))

   for W in [3, 6, 12, 24]:
       tmp[f"Fcst_power_rollmean{W}"] = tmp["Fcst_power"].rolling(W, min_periods=1).mean()
       tmp[f"Fcst_power_diff_rollmean{W}"] = tmp["Fcst_power_diff1"].rolling(W, min_periods=1).mean()

   eff_tmp = tmp.iloc[WINDOW_SIZE - 1:].reset_index(drop=True)

   time_eff0 = eff_tmp["Time"].values
   fc_eff0 = eff_tmp["Fcst"].values.astype(float)
   obs_eff0 = eff_tmp["Obs"].values.astype(float)
   n0 = len(eff_tmp)

   bias_eff0 = fc_eff0 - obs_eff0

   # ramp 相关标签
   P_obs_eff0 = windspeed_to_power(obs_eff0)
   obs_diffP0 = pd.Series(P_obs_eff0).diff().values
   ramp_flag_up0 = (obs_diffP0 >= RAMP_TH).astype(int)
   ramp_flag_down0 = (obs_diffP0 <= -RAMP_TH).astype(int)
   ramp_flag_obs0 = (np.abs(obs_diffP0) >= RAMP_TH).astype(int)

   ramp_flag_up0[0] = ramp_flag_down0[0] = ramp_flag_obs0[0] = 0

   feature_cols = [c for c in eff_tmp.columns if c not in ["Time", "Obs"]]
   X_lgb0 = eff_tmp[feature_cols].values

   # ========= Kalman Filter / EMA =========
   yt0 = bias_eff0
   Zt0 = np.column_stack([np.ones(n0), fc_eff0])
   _, bias_pred_kf0, _ = kalman_regression(yt0, Zt0, q=KF_Q, r=KF_R)
   fc_kf_eff0 = fc_eff0 - bias_pred_kf0

   bias_pred_EMA0 = EMA_bias_prediction(fc_eff0, obs_eff0, alpha=ALPHA_EMA)
   fc_EMA_eff0 = fc_eff0 - bias_pred_EMA0

   # ========= LSTM / TCN 特征准备 =========
   if n0 <= WINDOW_SIZE + 30:
       print(f"  有效样本 {n0} 不足训练深度模型，跳过该 Lead。")
       continue

   fc_diff1_eff = eff_tmp["Fcst_diff1"].fillna(0.0).values
   fc_diff_sign_eff = np.sign(fc_diff1_eff)

   P_fc_eff0 = windspeed_to_power(fc_eff0)
   diffP_fc_eff = np.diff(P_fc_eff0)
   diffP_fc_eff = np.concatenate([[0.0], diffP_fc_eff])
   diffP_fc_sign_eff = np.sign(diffP_fc_eff)

   features_seq_input = np.column_stack([
       fc_eff0,
       fc_diff1_eff,
       fc_diff_sign_eff,
       P_fc_eff0,
       diffP_fc_eff,
       diffP_fc_sign_eff,
   ])

   # === Strict scaler fit only on TRAIN period ===
   time_series_all = pd.to_datetime(time_eff0)
   train_mask_all0 = (time_series_all >= TRAIN_START) & (time_series_all <= TRAIN_END)

   if train_mask_all0.sum() < 50:
       print("  训练期数据太少，跳过该 Lead。")
       continue

   scaler_X = StandardScaler()
   scaler_X.fit(features_seq_input[train_mask_all0])
   features_scaled = scaler_X.transform(features_seq_input)

   scaler_y = StandardScaler()
   scaler_y.fit(bias_eff0[train_mask_all0].reshape(-1, 1))
   bias_scaled = scaler_y.transform(bias_eff0.reshape(-1, 1)).ravel()

   X_seq, y_seq = build_sequences(features_scaled, bias_scaled, WINDOW_SIZE)
   n_features_seq = features_scaled.shape[1]

   # === 时间映射回 sequence ===
   time_seq = time_eff0[WINDOW_SIZE - 1:]
   fc_seq = fc_eff0[WINDOW_SIZE - 1:]
   obs_seq = obs_eff0[WINDOW_SIZE - 1:]
   ramp_flag_obs_seq = ramp_flag_obs0[WINDOW_SIZE - 1:]
   ramp_flag_up_seq = ramp_flag_up0[WINDOW_SIZE - 1:]
   ramp_flag_down_seq = ramp_flag_down0[WINDOW_SIZE - 1:]

   fc_RAW_eff = fc_eff0[WINDOW_SIZE - 1:]
   fc_kf_eff = fc_kf_eff0[WINDOW_SIZE - 1:]
   fc_EMA_eff = fc_EMA_eff0[WINDOW_SIZE - 1:]
   X_lgb = X_lgb0[WINDOW_SIZE - 1:]
   bias_eff = bias_eff0[WINDOW_SIZE - 1:]

   time_series = pd.to_datetime(time_seq)

   # ========= 新时间分区 Train / Val / Test（严格保持你原结构） =========
   train_mask = (time_series >= TRAIN_START) & (time_series <= TRAIN_END)
   val_mask   = (time_series >= VAL_START)   & (time_series <= VAL_END)
   test_mask  = (time_series >= TEST_START)  & (time_series <= TEST_END)

   train_idx = np.where(train_mask)[0]
   val_idx   = np.where(val_mask)[0]
   test_idx  = np.where(test_mask)[0]

   if len(train_idx) < 50:
       print("训练集太少，跳过。")
       continue
   if len(val_idx) < 10:
       print("验证集太少，跳过。")
       continue
   if len(test_idx) < 10:
       print("测试集太少，跳过。")
       continue

   print(f"  训练集: {time_series[train_idx[0]]} ~ {time_series[train_idx[-1]]}  ({len(train_idx)})")
   print(f"  验证集: {time_series[val_idx[0]]} ~ {time_series[val_idx[-1]]}  ({len(val_idx)})")
   print(f"  测试集: {time_series[test_idx[0]]} ~ {time_series[test_idx[-1]]}  ({len(test_idx)})")

   # === ramp loss sample weights（保持你原写法） ===
   ramp_flag_up_train = ramp_flag_up_seq[train_idx]
   ramp_flag_down_train = ramp_flag_down_seq[train_idx]

   sample_weight_train = (
       1.0
       + RAMP_UP_WEIGHT * ramp_flag_up_train
       + RAMP_DOWN_WEIGHT * ramp_flag_down_train
   )

   # ========= Step 4: LSTM =========
   X_train_lstm = X_seq[train_idx]
   X_val_lstm = X_seq[val_idx]
   X_test_lstm = X_seq[test_idx]

   y_train_lstm = y_seq[train_idx]
   y_val_lstm = y_seq[val_idx]
   y_test_lstm = y_seq[test_idx]

   print("  → 训练 LSTM 偏差订正 ...")
   model_lstm = Sequential()
   model_lstm.add(LSTM(LSTM_UNITS, input_shape=(WINDOW_SIZE, n_features_seq)))
   model_lstm.add(Dense(1))
   model_lstm.compile(loss="mse", optimizer="adam")

   es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

   model_lstm.fit(
       X_train_lstm, y_train_lstm,
       validation_data=(X_val_lstm, y_val_lstm),
       epochs=EPOCHS,
       batch_size=BATCH_SIZE,
       callbacks=[es],
       sample_weight=sample_weight_train,
       verbose=0
   )

   y_pred_scaled_all_lstm = model_lstm.predict(X_seq, verbose=0).ravel()
   bias_pred_lstm_all = scaler_y.inverse_transform(
       y_pred_scaled_all_lstm.reshape(-1, 1)
   ).ravel()
   fc_lstm_eff = fc_seq - bias_pred_lstm_all

   # ========= Step 5: TCN =========
   print("  → 训练 TCN 偏差订正 ...")
   model_tcn = build_tcn_model(
       WINDOW_SIZE,
       n_features=n_features_seq,
       filters=TCN_FILTERS,
       kernel_size=TCN_KERNEL,
       dilations=(1, 2, 4, 8)
   )

   model_tcn.fit(
       X_train_lstm, y_train_lstm,
       validation_data=(X_val_lstm, y_val_lstm),
       epochs=EPOCHS,
       batch_size=BATCH_SIZE,
       callbacks=[es],
       sample_weight=sample_weight_train,
       verbose=0
   )

   y_pred_scaled_all_tcn = model_tcn.predict(X_seq, verbose=0).ravel()
   bias_pred_tcn_all = scaler_y.inverse_transform(
       y_pred_scaled_all_tcn.reshape(-1, 1)
   ).ravel()
   fc_tcn_eff = fc_seq - bias_pred_tcn_all

   # ========= Step 6: LightGBM =========
   X_train_lgb = X_lgb[train_idx]
   X_val_lgb = X_lgb[val_idx]

   y_train_lgb = bias_eff[train_idx]
   y_val_lgb = bias_eff[val_idx]

   print("  → LightGBM 偏差订正 ...")
   model_lgb = LGBMRegressor(
       n_estimators=1200,
       learning_rate=0.03,
       num_leaves=63,
       subsample=0.8,
       colsample_bytree=0.8,
       random_state=42,
       verbosity=-1,
   )

   model_lgb.fit(
       X_train_lgb, y_train_lgb,
       eval_set=[(X_val_lgb, y_val_lgb)],
       eval_metric="l2",
       sample_weight=sample_weight_train,
       callbacks=[early_stopping(stopping_rounds=50)]
   )

   bias_hat_lgb = model_lgb.predict(X_lgb)
   fc_lgb_eff = fc_seq - bias_hat_lgb

   # ========= Step 6.5: ramp skill（严格只用训练期）=========
   print("  → 计算 ramp skill（仅训练期 2021-09~2022-04）")

   ramp_skills = {}
   ramp_skills["LSTM"] = compute_ramp_skill_lstm(fc_lstm_eff, obs_seq, time_series, TEST_START, WINDOW_SIZE, RAMP_TH)
   ramp_skills["TCN"]  = compute_ramp_skill_lstm(fc_tcn_eff,  obs_seq, time_series, TEST_START, WINDOW_SIZE, RAMP_TH)
   ramp_skills["LGBM"] = compute_ramp_skill_lstm(fc_lgb_eff, obs_seq, time_series, TEST_START, WINDOW_SIZE, RAMP_TH)

   for k, v in ramp_skills.items():
       print(f"    模型 {k} ramp-skill={v:.3f}")

   # ========= Step 7: 多模型融合 =========
   cand_list_1 = [fc_kf_eff, fc_EMA_eff, fc_lstm_eff]
   w_hist_1, fc_fuse1_prior, _ = kalman_dynamic_fusion(
       cand_list_1, obs_seq, q_w=FUSION_QW, r_w=FUSION_RW
   )
   fc_opt1 = fc_fuse1_prior

   cand_list_2 = [fc_kf_eff, fc_EMA_eff, fc_lgb_eff]
   w_hist_2, fc_fuse2_prior, _ = kalman_dynamic_fusion(
       cand_list_2, obs_seq, q_w=FUSION_QW, r_w=FUSION_RW
   )
   fc_opt2 = fc_fuse2_prior

   cand_list_3 = [fc_kf_eff, fc_EMA_eff, fc_lstm_eff, fc_lgb_eff]
   w_hist_3, fc_fuse3_prior, _ = kalman_dynamic_fusion(
       cand_list_3, obs_seq, q_w=FUSION_QW, r_w=FUSION_RW
   )
   fc_opt3 = fc_fuse3_prior

   cand_list_5 = [fc_kf_eff, fc_EMA_eff, fc_lstm_eff, fc_tcn_eff]
   w_hist_5, fc_fuse5_prior, _ = kalman_dynamic_fusion(
       cand_list_5, obs_seq, q_w=FUSION_QW, r_w=FUSION_RW
   )
   fc_opt5 = fc_fuse5_prior

   # ========= Ramp 增强融合（保持你的结构）=========
   cand_names_4 = [
       "KF","EMA","LSTM","TCN","LGBM",
       "FUSE_KF_EMA_LSTM","FUSE_KF_EMA_LGBM",
       "FUSE_KF_EMA_LSTM_LGBM","FUSE_KF_EMA_LSTM_TCN"
   ]

   train_methods_sel = {
       "KF": fc_kf_eff,
       "EMA": fc_EMA_eff,
       "LSTM": fc_lstm_eff,
       "TCN": fc_tcn_eff,
       "LGBM": fc_lgb_eff,
       "FUSE_KF_EMA_LSTM": fc_opt1,
       "FUSE_KF_EMA_LGBM": fc_opt2,
       "FUSE_KF_EMA_LSTM_LGBM": fc_opt3,
       "FUSE_KF_EMA_LSTM_TCN": fc_opt5,
   }

   cand_list_4 = [train_methods_sel[name] for name in cand_names_4]
   name_to_idx = {name: i for i,name in enumerate(cand_names_4)}

   ramp_up_indices = []
   ramp_down_indices = []
   calm_pref_indices = [name_to_idx["KF"]]

   min_ramp_up_weight_indices = [name_to_idx["LSTM"]]
   min_ramp_down_weight_indices = [name_to_idx["LSTM"], name_to_idx["LGBM"]]

   # ramp flag from LSTM
   ramp_flag_pred = np.zeros_like(fc_lstm_eff, dtype=int)
   if len(fc_lstm_eff) > 1:
       P_fc = windspeed_to_power(fc_lstm_eff)
       P_obs_seq = windspeed_to_power(obs_seq)
       diffP_pred = P_fc[1:] - P_obs_seq[:-1]

       up_mask = diffP_pred >= RAMP_TH
       down_mask = diffP_pred <= -RAMP_TH

       ramp_flag_pred[1:][up_mask] = 1
       ramp_flag_pred[1:][down_mask] = 2

   w_hist_4, fc_fuse4_prior, _ = kalman_dynamic_fusion(
       cand_list_4, obs_seq,
       q_w=FUSION_QW, r_w=FUSION_RW,
       ramp_flags=ramp_flag_pred,
       ramp_up_indices=ramp_up_indices,
       ramp_down_indices=ramp_down_indices,
       calm_pref_indices=calm_pref_indices,
       ramp_up_boost_factor=RAMP_UP_BOOST_FACTOR,
       ramp_down_boost_factor=RAMP_DOWN_BOOST_FACTOR,
       calm_boost_factor=CALM_BOOST_FACTOR,
       min_ramp_up_weight_indices=min_ramp_up_weight_indices,
       min_ramp_down_weight_indices=min_ramp_down_weight_indices,
       min_weight_up=0.9,
       min_weight_down=0.45,
   )

   fc_opt4 = fc_fuse4_prior

   print("  动态融合1(KF+EMA+LSTM) 平均权重：", ", ".join(f"{w:.3f}" for w in w_hist_1.mean(axis=0)))
   print("  动态融合2(KF+EMA+LGBM) 平均权重：", ", ".join(f"{w:.3f}" for w in w_hist_2.mean(axis=0)))
   print("  动态融合3(KF+EMA+LSTM+LGBM) 平均权重：", ", ".join(f"{w:.3f}" for w in w_hist_3.mean(axis=0)))
   print("  动态融合5(KF+EMA+LSTM+TCN) 平均权重：", ", ".join(f"{w:.3f}" for w in w_hist_5.mean(axis=0)))
   print("  动态融合4(RAMP增强+筛选) 平均权重：", ", ".join(f"{w:.3f}" for w in w_hist_4.mean(axis=0)))

   print_fusion_weight_by_ramp(
       w_hist=w_hist_4,
       ramp_flags=ramp_flag_pred,
       model_names=cand_names_4,
       lead_hour=lead_hour,
       fusion_name="FUSE_RAMP_ENHANCED"
   )

   # ========= Step 7.9: 汇总各方案 =========
   methods = {
       "RAW":                       fc_RAW_eff,
       "KF":                        fc_kf_eff,
       "EMA":                       fc_EMA_eff,
       "LSTM":                      fc_lstm_eff,
       "TCN":                       fc_tcn_eff,
       "LGBM":                      fc_lgb_eff,
       "FUSE_KF_EMA_LSTM":         fc_opt1,
       "FUSE_KF_EMA_LGBM":         fc_opt2,
       "FUSE_KF_EMA_LSTM_LGBM":    fc_opt3,
       "FUSE_KF_EMA_LSTM_TCN":     fc_opt5,
       "FUSE_RAMP_ENHANCED":       fc_opt4,
   }

   obs_test = obs_seq[test_idx]
   RAW_test = fc_RAW_eff[test_idx]

   # ========= Step 8: 测试指标（2024-03~12） =========
   stats_test = {}
   ramp_acc_test = {}
   for name, series in methods.items():
       mae, mse, rmse = calc_stats(series[test_idx], obs_test)
       corr = calc_corr(series[test_idx], obs_test)
       stats_test[name] = (mae, rmse, corr)
       ramp_acc_test[name] = compute_ramp_accuracy(series[test_idx], obs_test, RAMP_TH)

   fuse_test = fc_opt4[test_idx]

   # ========= Step 8.5: 3 类 Ramp 混淆矩阵 =========
   CM_METHODS_TO_PLOT = [
       "RAW",
       "KF",
       "EMA",
       "LSTM",
       "TCN",
       "LGBM",
       "FUSE_RAMP_ENHANCED",
   ]

   from math import ceil

   obs_label_3_test = compute_ramp_label_3class(obs_test, RAMP_TH)
   cm_dict = {}
   labels = [0, 1, 2]

   for name, series in methods.items():
       if name not in CM_METHODS_TO_PLOT:
           continue
       series_test = series[test_idx]
       fc_label_3_test = compute_ramp_label_3class(series_test, RAMP_TH)
       cm_dict[name] = confusion_matrix(obs_label_3_test, fc_label_3_test, labels=labels)

   metrics_dict = {}
   y_true = obs_label_3_test

   print(f"\n  [测试集三分类指标 | Lead={lead_hour}h | 测试=2024-03~12]")
   for name, series in methods.items():
       if name not in CM_METHODS_TO_PLOT:
           continue
       y_pred = compute_ramp_label_3class(series[test_idx], RAMP_TH)
       m = compute_multi_metrics(y_true, y_pred)
       metrics_dict[name] = m

       print(
           f"    {name:24s} "
           f"Acc={m['acc']:.3f}  Macro(P/R/F1)=({m['macro_p']:.3f}/{m['macro_r']:.3f}/{m['macro_f1']:.3f})  "
           f"Up(P/R/F1)=({m['p1']:.3f}/{m['r1']:.3f}/{m['f11']:.3f})  "
           f"Down(P/R/F1)=({m['p2']:.3f}/{m['r2']:.3f}/{m['f12']:.3f})"
       )

   if cm_dict:
       method_names = list(cm_dict.keys())
       n_methods = len(method_names)
       n_cols = 3
       n_rows = ceil(n_methods / n_cols)

       fig, axes = plt.subplots(
           n_rows, n_cols,
           figsize=(4.5 * n_cols, 4 * n_rows),
           squeeze=False
       )

       class_names = ["No-Ramp", "Ramp-Up", "Ramp-Down"]

       fontsize_title = 19
       fontsize_axis = 19
       fontsize_tick = 13.5
       fontsize_number = 20

       used_positions = set()

       last_row_count = n_methods % n_cols
       if last_row_count == 0:
           last_row_count = n_cols
       start_col_last_row = (n_cols - last_row_count) // 2 if last_row_count < n_cols else 0

       for idx, name in enumerate(method_names):
           r = idx // n_cols
           c = idx % n_cols

           if r == n_rows - 1 and last_row_count < n_cols:
               c = start_col_last_row + c

           used_positions.add((r, c))
           ax = axes[r, c]
           cm = cm_dict[name]

           im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
           letter = f"({chr(ord('a') + idx)})"
           ax.set_title(f"{letter}{name}", fontsize=fontsize_title)

           ax.set_xlabel("Predicted Label", fontsize=fontsize_axis)
           ax.set_ylabel("True Label", fontsize=fontsize_axis)

           ax.set_xticks(range(3))
           ax.set_yticks(range(3))
           ax.set_xticklabels(class_names, rotation=0, fontsize=fontsize_tick)
           ax.set_yticklabels(class_names, fontsize=fontsize_tick)

           thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
           for i in range(3):
               for j in range(3):
                   ax.text(j, i, cm[i, j],
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=fontsize_number)

       for r in range(n_rows):
           for c in range(n_cols):
               if (r, c) not in used_positions:
                   axes[r, c].axis("off")

       fig.suptitle(f"Lead={lead_hour}h Ramp Confusion Matrix (Test=2024-03~12)", fontsize=28)

       fig.subplots_adjust(
           top=0.85,
           bottom=0.08,
           left=0.06,
           right=0.88,
           hspace=0.40,
           wspace=0.70
       )

       cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.30])
       cbar = fig.colorbar(im, cax=cbar_ax)
       cbar.ax.tick_params(labelsize=18)

       out_cm_all_name = f"Ramp_CM_Lead_{lead_hour}h_ALL_MODELS.png"
       ##save_png_and_eps(out_cm_all_name, dpi=350, fig=fig)

       ##plt.show()
       plt.close(fig)

   print("  [测试集指标（全部为前验预测）]：")
   for name, (mae, rmse, corr) in stats_test.items():
       print(f"    {name:24s} RMSE={rmse:.3f}, MAE={mae:.3f}, Corr={corr:.3f}")

   print("  [测试集 Ramp 识别准确率]：")
   for name, acc in ramp_acc_test.items():
       print(f"    {name:24s} RampAcc={acc:.3f}")

   # ========= Step *: 画图 =========
   t_test = time_series[test_idx]

   """
   fig1_name = f"Timeseries_Lead_{lead_hour}h_TEST_KF_EMA_LSTM.png"
   plot_obs_and_fuse_with_ramps(
       time_test=t_test,
       obs_test=obs_test,
       fc_test=fc_opt1[test_idx],
       fc_RAW_test=RAW_test,
       title=f"Lead={lead_hour}h 测试(2024-03~12)：KF+EMA+LSTM 融合 vs 实测",
       save_name=fig1_name,
       ramp_th=RAMP_TH
   )

   fig2_name = f"Timeseries_Lead_{lead_hour}h_TEST_KF_EMA_LGBM.png"
   plot_obs_and_fuse_with_ramps(
       time_test=t_test,
       obs_test=obs_test,
       fc_test=fc_opt2[test_idx],
       fc_RAW_test=RAW_test,
       title=f"Lead={lead_hour}h 测试(2024-03~12)：KF+EMA+LGBM 融合 vs 实测",
       save_name=fig2_name,
       ramp_th=RAMP_TH
   )

   fig3_name = f"Timeseries_Lead_{lead_hour}h_TEST_KF_EMA_LSTM_LGBM.png"
   plot_obs_and_fuse_with_ramps(
       time_test=t_test,
       obs_test=obs_test,
       fc_test=fc_opt3[test_idx],
       fc_RAW_test=RAW_test,
       title=f"Lead={lead_hour}h 测试(2024-03~12)：KF+EMA+LSTM+LGBM 融合 vs 实测",
       save_name=fig3_name,
       ramp_th=RAMP_TH
   )

   fig5_name = f"Timeseries_Lead_{lead_hour}h_TEST_KF_EMA_LSTM_TCN.png"
   plot_obs_and_fuse_with_ramps(
       time_test=t_test,
       obs_test=obs_test,
       fc_test=fc_opt5[test_idx],
       fc_RAW_test=RAW_test,
       title=f"Lead={lead_hour}h 测试(2024-03~12)：KF+EMA+LSTM+TCN 融合 vs 实测",
       save_name=fig5_name,
       ramp_th=RAMP_TH
   )

   fig4_name = f"Timeseries_Lead_{lead_hour}h_TEST_FUSE_RAMP_ENHANCED.png"
   plot_obs_and_fuse_with_ramps(
       time_test=t_test,
       obs_test=obs_test,
       fc_test=fuse_test,
       fc_RAW_test=RAW_test,
       title=f"Lead={lead_hour}h 测试(2024-03~12)：FUSE_RAMP_ENHANCED vs 实测",
       save_name=fig4_name,
       ramp_th=RAMP_TH
   )
    """
   # ========= Step 10: 保存当前 lead Excel =========
   out_name = f"KFE_LSTM_TCN_LGBM_FUSION_RAMP_Train2023_Val20240102_Test20240312_Lead_{lead_hour}h.xlsx"
   out_df = pd.DataFrame({
       "Time": time_series,
       "Obs": obs_seq,
       "Fcst_RAW": fc_RAW_eff,
       "Fcst_KF": fc_kf_eff,
       "Fcst_EMA": fc_EMA_eff,
       "Fcst_LSTM": fc_lstm_eff,
       "Fcst_TCN": fc_tcn_eff,
       "Fcst_LGBM": fc_lgb_eff,
       "Fcst_FUSE_KF_EMA_LSTM":        fc_opt1,
       "Fcst_FUSE_KF_EMA_LGBM":        fc_opt2,
       "Fcst_FUSE_KF_EMA_LSTM_LGBM":   fc_opt3,
       "Fcst_FUSE_KF_EMA_LSTM_TCN":    fc_opt5,
       "Fcst_FUSE_RAMP_ENHANCED":      fc_opt4,
   })

   out_df["is_train"] = 0
   out_df["is_val"] = 0
   out_df["is_test"] = 0
   out_df.loc[train_idx, "is_train"] = 1
   out_df.loc[val_idx, "is_val"] = 1
   out_df.loc[test_idx, "is_test"] = 1

   with pd.ExcelWriter(out_name) as writer:
       out_df.to_excel(writer, sheet_name="TimeSeries", index=False)

       stats_rows = []
       for name, (mae, rmse, corr) in stats_test.items():
           stats_rows.append({"方案": name, "MAE": mae, "RMSE": rmse, "Corr": corr})
       pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Stats_TEST", index=False)

       ramp_rows = []
       for name, acc in ramp_acc_test.items():
           ramp_rows.append({"方案": name, "Ramp_Acc": acc})
       pd.DataFrame(ramp_rows).to_excel(writer, sheet_name="RampAcc_TEST", index=False)

       metrics_rows = []
       for name, m in metrics_dict.items():
           metrics_rows.append({
               "方案": name,
               "Acc": m["acc"],
               "Macro_P": m["macro_p"], "Macro_R": m["macro_r"], "Macro_F1": m["macro_f1"],
               "P_NoRamp": m["p0"], "R_NoRamp": m["r0"], "F1_NoRamp": m["f10"],
               "P_RampUp": m["p1"], "R_RampUp": m["r1"], "F1_RampUp": m["f11"],
               "P_RampDown": m["p2"], "R_RampDown": m["r2"], "F1_RampDown": m["f12"],
           })
       metrics_df = pd.DataFrame(metrics_rows)
       metrics_df.to_excel(writer, sheet_name="RampMetrics_TEST", index=False)

   print(f"  → 已保存 Excel：{out_name}")

   # ========= 汇总表行 =========
   row = {
       "Lead_h": lead_hour,

       "RMSE_RAW":  stats_test["RAW"][1],
       "RMSE_KF":   stats_test["KF"][1],
       "RMSE_EMA":  stats_test["EMA"][1],
       "RMSE_LSTM": stats_test["LSTM"][1],
       "RMSE_TCN":  stats_test["TCN"][1],
       "RMSE_LGBM": stats_test["LGBM"][1],
       "RMSE_FUSE_KF_EMA_LSTM":        stats_test["FUSE_KF_EMA_LSTM"][1],
       "RMSE_FUSE_KF_EMA_LGBM":        stats_test["FUSE_KF_EMA_LGBM"][1],
       "RMSE_FUSE_KF_EMA_LSTM_LGBM":   stats_test["FUSE_KF_EMA_LSTM_LGBM"][1],
       "RMSE_FUSE_KF_EMA_LSTM_TCN":    stats_test["FUSE_KF_EMA_LSTM_TCN"][1],
       "RMSE_FUSE_RAMP_ENHANCED":      stats_test["FUSE_RAMP_ENHANCED"][1],

       "MAE_RAW":  stats_test["RAW"][0],
       "MAE_KF":   stats_test["KF"][0],
       "MAE_EMA":  stats_test["EMA"][0],
       "MAE_LSTM": stats_test["LSTM"][0],
       "MAE_TCN":  stats_test["TCN"][0],
       "MAE_LGBM": stats_test["LGBM"][0],
       "MAE_FUSE_KF_EMA_LSTM":        stats_test["FUSE_KF_EMA_LSTM"][0],
       "MAE_FUSE_KF_EMA_LGBM":        stats_test["FUSE_KF_EMA_LGBM"][0],
       "MAE_FUSE_KF_EMA_LSTM_LGBM":   stats_test["FUSE_KF_EMA_LSTM_LGBM"][0],
       "MAE_FUSE_KF_EMA_LSTM_TCN":    stats_test["FUSE_KF_EMA_LSTM_TCN"][0],
       "MAE_FUSE_RAMP_ENHANCED":      stats_test["FUSE_RAMP_ENHANCED"][0],

       "Corr_RAW":  stats_test["RAW"][2],
       "Corr_KF":   stats_test["KF"][2],
       "Corr_EMA":  stats_test["EMA"][2],
       "Corr_LSTM": stats_test["LSTM"][2],
       "Corr_TCN":  stats_test["TCN"][2],
       "Corr_LGBM": stats_test["LGBM"][2],
       "Corr_FUSE_KF_EMA_LSTM":        stats_test["FUSE_KF_EMA_LSTM"][2],
       "Corr_FUSE_KF_EMA_LGBM":        stats_test["FUSE_KF_EMA_LGBM"][2],
       "Corr_FUSE_KF_EMA_LSTM_LGBM":   stats_test["FUSE_KF_EMA_LSTM_LGBM"][2],
       "Corr_FUSE_KF_EMA_LSTM_TCN":    stats_test["FUSE_KF_EMA_LSTM_TCN"][2],
       "Corr_FUSE_RAMP_ENHANCED":      stats_test["FUSE_RAMP_ENHANCED"][2],

       "RampAcc_RAW":                  ramp_acc_test["RAW"],
       "RampAcc_KF":                   ramp_acc_test["KF"],
       "RampAcc_EMA":                  ramp_acc_test["EMA"],
       "RampAcc_LSTM":                 ramp_acc_test["LSTM"],
       "RampAcc_TCN":                  ramp_acc_test["TCN"],
       "RampAcc_LGBM":                 ramp_acc_test["LGBM"],
       "RampAcc_FUSE_KF_EMA_LSTM":     ramp_acc_test["FUSE_KF_EMA_LSTM"],
       "RampAcc_FUSE_KF_EMA_LGBM":     ramp_acc_test["FUSE_KF_EMA_LGBM"],
       "RampAcc_FUSE_KF_EMA_LSTM_LGBM": ramp_acc_test["FUSE_KF_EMA_LSTM_LGBM"],
       "RampAcc_FUSE_KF_EMA_LSTM_TCN":  ramp_acc_test["FUSE_KF_EMA_LSTM_TCN"],
       "RampAcc_FUSE_RAMP_ENHANCED":   ramp_acc_test["FUSE_RAMP_ENHANCED"],
   }

   summary_rows.append(row)

   for name, m in metrics_dict.items():
       ramp_metrics_summary_rows.append({
           "Lead_h": lead_hour,
           "方案": name,
           "Acc": m["acc"],
           "Macro_P": m["macro_p"], "Macro_R": m["macro_r"], "Macro_F1": m["macro_f1"],
           "P_NoRamp": m["p0"], "R_NoRamp": m["r0"], "F1_NoRamp": m["f10"],
           "P_RampUp": m["p1"], "R_RampUp": m["r1"], "F1_RampUp": m["f11"],
           "P_RampDown": m["p2"], "R_RampDown": m["r2"], "F1_RampDown": m["f12"],
       })


# ==============================
# 所有 Lead 汇总
# ==============================
if summary_rows:
    import matplotlib as mpl

    # =============== 全局学术风格 ===============
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],

        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        "axes.linewidth": 0.9,
        "lines.linewidth": 1.6,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        "axes.grid": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,

        "legend.frameon": False,

        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    # ===========================================================
    #                Summary Table & Basic Data
    # ===========================================================
    summary_df = pd.DataFrame(summary_rows).sort_values("Lead_h")
    summary_df.to_excel(
        "KFE_LSTM_TCN_LGBM_FUSION_RAMP_5train1test_Lead_Summary.xlsx",
        index=False
    )
    print("\n=== Lead-wise Test Metrics Summary (All One-step Ahead) ===")
    print(summary_df)

    x = summary_df["Lead_h"].values

    # Academic color palette
    colors = {
        "RAW":   "#1f77b4",
        "KF":    "#ff7f0e",
        "EMA":   "#2ca02c",
        "LSTM":  "#d62728",
        "TCN":   "#9467bd",
        "LGBM":  "#8c564b",
        "FUSE":  "#000000",
    }

    # ===========================================================
    #      单个子图：在指定 ax 上画一类 metric 的多曲线
    # ===========================================================
    def plot_metric_panel(ax, x, df, curves, ylabel, title):
        """
        ax: 子图坐标轴
        curves: list of (col, label, color_key, marker, is_main)
        """
        for col, lab, col_code, m, is_main in curves:
            y = df[col].values

            z = 3 if is_main else 2
            lw = 2.2 if is_main else 1.6
            ms = 26 if is_main else 18
            mew = 0.9 if is_main else 0.8

            # 连线
            ax.plot(
                x,
                y,
                color=colors[col_code],
                linewidth=lw,
                label=lab,
                zorder=z,
            )
            # 标记：白芯 + 有色边框
            ax.scatter(
                x,
                y,
                marker=m,
                s=ms,
                facecolors="white",
                edgecolors=colors[col_code],
                linewidths=mew,
                zorder=z + 0.5,
            )

        ax.set_xlabel("Lead Time (h)", labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_title(title, pad=3)

        # 只保留下/左边框
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(0.9)

        # 整数 lead
        ax.set_xticks(x)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", pad=2)
        ax.tick_params(axis="both", which="minor", pad=2)

        # 轻微 y 轴网格
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.35)

    # ===========================================================
    #         创建 2×2 子图：3 个 metric + 1 个图注
    # ===========================================================
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.8))  # 接近正方形

    ax_rmse   = axes[0, 0]
    ax_mae    = axes[0, 1]
    ax_corr   = axes[1, 0]
    ax_legend = axes[1, 1]

    # ------------------- 1) RMSE 面板 -------------------
    plot_metric_panel(
        ax_rmse,
        x,
        summary_df,
        curves=[
            ("RMSE_RAW",                "RAW",   "RAW",   "o", False),
            ("RMSE_KF",                 "KF",    "KF",    "s", False),
            ("RMSE_EMA",                "EMA",   "EMA",   "^", False),
            ("RMSE_LSTM",               "LSTM",  "LSTM",  "P", False),
            ("RMSE_TCN",                "TCN",   "TCN",   "v", False),
            ("RMSE_LGBM",               "LGBM",  "LGBM",  "d", False),
            ("RMSE_FUSE_RAMP_ENHANCED", "FUSE_RAMP_ENHANCED", "FUSE", "*", True),
        ],
        ylabel="RMSE (m/s)",
        title="(a) RMSE",
    )

    # ------------------- 2) MAE 面板 -------------------
    plot_metric_panel(
        ax_mae,
        x,
        summary_df,
        curves=[
            ("MAE_RAW",                "RAW",   "RAW",   "o", False),
            ("MAE_KF",                 "KF",    "KF",    "s", False),
            ("MAE_EMA",                "EMA",   "EMA",   "^", False),
            ("MAE_LSTM",               "LSTM",  "LSTM",  "P", False),
            ("MAE_TCN",                "TCN",   "TCN",   "v", False),
            ("MAE_LGBM",               "LGBM",  "LGBM",  "d", False),
            ("MAE_FUSE_RAMP_ENHANCED", "FUSE_RAMP_ENHANCED", "FUSE", "*", True),
        ],
        ylabel="MAE (m/s)",
        title="(b) MAE",
    )

    # ------------------- 3) Corr 面板 -------------------
    plot_metric_panel(
        ax_corr,
        x,
        summary_df,
        curves=[
            ("Corr_RAW",                "RAW",   "RAW",   "o", False),
            ("Corr_KF",                 "KF",    "KF",    "s", False),
            ("Corr_EMA",                "EMA",   "EMA",   "^", False),
            ("Corr_LSTM",               "LSTM",  "LSTM",  "P", False),
            ("Corr_TCN",                "TCN",   "TCN",   "v", False),
            ("Corr_LGBM",               "LGBM",  "LGBM",  "d", False),
            ("Corr_FUSE_RAMP_ENHANCED", "FUSE_RAMP_ENHANCED", "FUSE", "*", True),
        ],
        ylabel="Correlation Coefficient",
        title="(c) Correlation",
    )

    # ------------------- 4) 图注面板 -------------------
    ax_legend.axis("off")  # 不画坐标轴，只放 legend

    # 从其中一幅图拿到所有 handle / label
    handles, labels = ax_rmse.get_legend_handles_labels()

    # 居中放 legend，可以按需要调整 ncol
    leg = ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=2,                 # 或者 3，看你线条多少
        handlelength=2.2,
        columnspacing=1.2,
        borderaxespad=0.3,
        frameon=False,
        title="Methods",
        title_fontsize=8,
    )

    # 调整子图间距
    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.0)

    # 统一保存成一张总图
    save_png_and_eps(
        "KFE_FUSION_RAMP_5train1test_Metrics_vs_Lead_Combined.png",
        dpi=600
    )

# ===============================================================
#                 Ramp Metrics Summary (English)
# ===============================================================
if ramp_metrics_summary_rows:
    metrics_summary_df = pd.DataFrame(ramp_metrics_summary_rows)
    metrics_summary_df = metrics_summary_df.sort_values(["Lead_h", "方案"])
    metrics_summary_df.to_excel(
        "KFE_LSTM_TCN_LGBM_FUSION_RAMP_5train1test_RampMetrics_Summary.xlsx",
        index=False
    )
    print("\n=== Lead-wise Ramp Classification Metrics (Acc / Macro P-R-F1 / Per-Class P-R-F1) ===")
    print(metrics_summary_df)

plt.show()
print("\nAll figures generated successfully. ✅")
