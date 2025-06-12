from rc_pulse_transit_time import PTTPPGData
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Get the files
files = glob.glob('../DATA/csv/*sit.csv')

info_file = '../DATA/info/subjects_info.csv'
df = pd.read_csv(info_file)

# keep only sitting
df = df.loc[df.activity == 'sit', :].copy().reset_index(drop=True)
df['BMI'] = df.weight/df.height**2
gender_num = np.zeros(len(df))
gender_num[df.gender == 'female'] = 1
df['gender_num'] = gender_num

# Get demographic composite feature with PCA
df['DEM'] = PCA(n_components=1).fit_transform(df[['BMI', 'age', 'height', 'weight', 'gender_num']])

# For each files, get the lag value between pleth at distal and proximal phalanx
df['pleth_lag'] = np.zeros(len(df))
df['bpm'] = np.zeros(len(df))
df['bpm_pleth'] = np.zeros(len(df))
features = None
for f in tqdm(files, f'Processing files', total=len(files)):
    data = PTTPPGData(f)
    data.clean_data()
    data.compute_ppg_peaks()
    lag = data.get_phalanx_lag()
    bpm = data.get_ecg_hr()
    bpm_pleth = data.get_hr()
    df.loc[df.record==f'{data.subject}_{data.activity}', 'pleth_lag'] = lag
    df.loc[df.record == f'{data.subject}_{data.activity}', 'bpm'] = bpm
    df.loc[df.record == f'{data.subject}_{data.activity}', 'bpm_pleth'] = bpm_pleth

    features = data.get_average_peak_feature(remove_outlier=True)
    fe_freq = data.get_frequency_features()
    features.update(fe_freq)

    for k in features.keys():
        df.loc[df.record == f'{data.subject}_{data.activity}', k] = features[k]


# Start feature preparation
# Try a PPG composite with PCA since there's not enough data to use all features
pca = PCA(n_components=3)
pc = pca.fit_transform(df[features.keys()])
df['PPG1'] = pc[:, 0]
df['PPG2'] = pc[:, 1]
feature_names = ['DEM', 'pleth_lag', 'bpm', 'PPG1', 'PPG2']
X = df.loc[:, feature_names].copy()
Y_sys = df['bp_sys_end'].values
Y_dia = df['bp_dia_end'].values

# Simple linear regression to assess the data with the whole dataset
xo = sm.add_constant(X)
print(sm.OLS(Y_sys, xo).fit().summary())

# Regression with leave one out, using Lasso as there's not a lot of datapoints, try with and without PPG feature
Y = Y_sys
for fg in [['DEM', 'bpm', 'PPG1', 'PPG2'],
           ['DEM', 'bpm', 'PPG1', 'PPG2', 'pleth_lag'],
           ['DEM', 'bpm']]:
    X = df.loc[:, fg].copy()
    model = make_pipeline(StandardScaler(), SVR(epsilon=0.2, kernel='linear'))
    y_pred = np.zeros_like(Y)
    y_real = np.zeros_like(Y)
    loo = LeaveOneOut()
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(f"Fold {i}:")
        x_train = X.loc[train_index, :]
        y_train = Y[train_index]
        model.fit(x_train, y_train)
        y_pred[i] = model.predict(X.loc[test_index, :])[0]
        y_real[i] = Y[test_index[0]]

    # Result
    plt.figure()
    r2 = r2_score(y_real, y_pred)
    rho, pv = pearsonr(y_real, y_pred)
    sns.regplot(x=y_real, y=y_pred)
    plt.gca().set_aspect('equal')
    plt.gca().set_ylim(80, 150)
    plt.gca().set_xlim(80, 150)
    plt.plot([80, 150], [80, 150])
    plt.xlabel('Real Systolic pressure')
    plt.ylabel('Predicted Systolic pressure')
    if 'PPG1' in fg:
        if 'pleth_lag' in fg:
            plt.title(f'SVR with DEM, BPM, PPG and pleth lag\nR2 = {r2:.2f} - rho = {rho:.2f} - p = {pv:.4f}')
            plt.savefig('../figures/results_svr_withPPG_plethLag.jpg')
        else:
            plt.title(f'SVR with DEM, BPM and PPG\nR2 = {r2:.2f} - rho = {rho:.2f} - p = {pv:.4f}')
            plt.savefig('../figures/results_svr_withPPG.jpg')
    else:
        plt.title(f'SVR with DEM, without PPG\nR2 = {r2:.2f} - rho = {rho:.2f} - p = {pv:.4f}')
        plt.savefig('../figures/results_svr_withOutPPG.jpg')

    plt.show()
