import glob
from rc_pulse_transit_time import PTTPPGData
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

files = glob.glob('../DATA/csv/*.csv')
hr = []
hr_ple = []
for f in files:
    # if 's12' in f:
        pdata = PTTPPGData(f)
        pdata.clean_data()
        hr_ple.append(pdata.get_hr())
        hr.append(pdata.get_ecg_hr())
        print(f'{pdata.subject}, hr {hr[-1]}, phr {hr_ple[-1]}')

plt.figure()
plt.scatter(hr, hr_ple)
plt.gca().set_aspect('equal')
plt.gca().set_ylim(20, 90)
plt.gca().set_xlim(20, 90)
rho, pv = pearsonr(hr, hr_ple)
plt.title(f'rho = {rho:.3f}')
plt.show()
