from ppt_ppg_data import PPTPPGData
import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# Get the files
files = glob.glob('../DATA/csv/*.csv')

info_file = '../DATA/info/subjects_info.csv'
df = pd.read_csv(info_file)

# For each files, get the lag value between pleth at distal and proximal phalanx
df['pleth_lag'] = np.zeros(len(df))
df['bpm'] = np.zeros(len(df))
df['stt'] = np.zeros(len(df))
for f in files:
    data = PPTPPGData(f)
    data.clean_data()
    lag = data.get_phalanx_lag()
    bpm = data.get_ecg_hr()
    stt = data.get_stt()

    df.loc[df.record==f'{data.subject}_{data.activity}', 'pleth_lag'] = lag
    df.loc[df.record == f'{data.subject}_{data.activity}', 'bpm'] = bpm
    df.loc[df.record == f'{data.subject}_{data.activity}', 'stt'] = stt

df['gender_num'] = np.zeros(len(df))
df.loc[df.gender == 'male', 'gender_num'] = 1
feature_names = ['height', 'weight', 'age', 'pleth_lag', 'stt']

X = df.loc[df.activity=='sit', feature_names].copy().reset_index(drop=True)
# Y = df.bp_sys_start[df.activity=='sit'].values - df.bp_dia_start[df.activity=='sit'].values
Y = df.bp_sys_end[df.activity=='sit'].values

model = AdaBoostRegressor(n_estimators=300)

y_pred = np.zeros_like(Y)
loo = LeaveOneOut()
for i, (train_index, test_index) in enumerate(loo.split(X)):
    print(f"Fold {i}:")
    x_train = X.loc[train_index, :]
    y_train = Y[train_index]

    model.fit(x_train, y_train)
    y_pred[test_index] = model.predict(X.loc[test_index, :])

r2 = r2_score(Y, y_pred)
sns.regplot(x=Y, y=y_pred)
plt.show()
