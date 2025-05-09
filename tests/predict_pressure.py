from rc_pulse_transit_time import PTTPPGData
import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict

# Get the files
files = glob.glob('../DATA/csv/*sit.csv')

info_file = '../DATA/info/subjects_info.csv'
df = pd.read_csv(info_file)
df['BMI'] = df.weight/df.height**2

# For each files, get the lag value between pleth at distal and proximal phalanx
df['pleth_lag'] = np.zeros(len(df))
df['bpm'] = np.zeros(len(df))
features = None
for f in tqdm(files, f'Processing files', total=len(files)):
    data = PTTPPGData(f)
    data.clean_data()
    data.compute_ppg_peaks()
    lag = data.get_phalanx_lag()
    bpm = data.get_ecg_hr()

    df.loc[df.record==f'{data.subject}_{data.activity}', 'pleth_lag'] = lag
    df.loc[df.record == f'{data.subject}_{data.activity}', 'bpm'] = bpm


    features = data.get_average_peak_feature()
    fe_freq = data.get_frequency_features()
    features.update(fe_freq)

    for k in features.keys():
        df.loc[df.record == f'{data.subject}_{data.activity}', k] = features[k]


feature_names = ['age', 'height', 'weight', 'BMI', 'pleth_lag', 'bpm'] + list(features.keys())

X = df.loc[df.activity=='sit', feature_names].copy().reset_index(drop=True)
Y = df.loc[df.activity=='sit', ['bp_sys_start', 'bp_sys_end']].mean(axis=1).values

# Now remove collinearity until n features
corr = spearmanr(X).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
cluster_ids = hierarchy.fcluster(dist_linkage, 8, criterion="maxclust")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

X = X.iloc[:, selected_features]
# Classification

rfecv = RFECV(
    estimator=AdaBoostRegressor(),
    step=1,
    cv=RepeatedKFold(n_splits=4, n_repeats=5),
    scoring='r2',
    min_features_to_select=8,
    n_jobs=2,
)

scaler = StandardScaler()
model = AdaBoostRegressor(n_estimators=300)

pipe = Pipeline([
    ('scaler', scaler),
    # ('feature selection', rfecv),
    ('adaboost', model)
])

y_pred = np.zeros_like(Y)
loo = LeaveOneOut()
for i, (train_index, test_index) in enumerate(loo.split(X)):
    print(f"Fold {i}:")
    x_train = X.loc[train_index, :]
    y_train = Y[train_index]

    pipe.fit(x_train, y_train)
    y_pred[test_index] = pipe.predict(X.loc[test_index, :])

    # model.fit(x_train, y_train)
    # y_pred[test_index] = model.predict(X.loc[test_index, :])

plt.figure()
r2 = r2_score(Y, y_pred)
sns.regplot(x=Y, y=y_pred)
plt.gca().set_aspect('equal')
plt.gca().set_ylim(80, 150)
plt.gca().set_xlim(80, 150)
plt.plot([80, 150], [80, 150])
plt.xlabel('Real Systolic pressure')
plt.ylabel('Predicted Systolic pressure')
plt.title(f'R2 = {r2}')
plt.show()
