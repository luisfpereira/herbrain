import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import statsmodels.api as sm
from pathlib import Path
from sklearn.cross_decomposition import CCA
from support.kernels.torch_kernel import TorchKernel

import herbrain.lddmm as lddmm
import herbrain.lddmm.strings as strings
from herbrain.depression.hotelling import hotelling_t2
from herbrain.parallel_transport import main as parallel_transport


output_dir = Path()

# compute atlas from control group at t0
data_set = [
    {'shape': 'path/to/data/control/subject1/t0'},
    {'shape': 'path/to/data/control/subject2/t0'},
]

# fine tune depending on data
kernel_width = 4
registration_args = dict(
    kernel_width=kernel_width, regularisation=1, max_iter=2000,
    freeze_control_points=False, metric='varifold', attachment_kernel_width=2.,
    tol=1e-10, filter_cp=True, threshold=0.75)

n_rungs = 10
transport_args = {
    'kernel_type': 'torch', 'kernel_width': kernel_width, 'n_rungs': n_rungs}

shoot_args = {
    'use_rk2_for_flow': True, 'kernel_width': kernel_width,
    'number_of_time_steps': n_rungs + 1, 'write_params': False}


atlas_dir = output_dir / 'atlas'
lddmm.deterministic_atlas(
    data_set[0]['shape'], data_set, 'structure', atlas_dir,
    initial_step_size=1e-1, **registration_args)
atlas_cp_path = atlas_dir / strings.cp_str
atlas = atlas_dir / 'atlas'

atlas_cp = lddmm.read_2D_array(atlas_cp_path)
n_cp = atlas_cp.shape[-1]
kernel = TorchKernel(kernel_width=kernel_width, device='auto')

# parallel transport
subjects = []
transport_result = pd.DataFrame()
for name in subjects:
    source = f'path/to/data/{name}/t0'
    target = f'path/to/data/{name}/t1'
    shoot_args['source'] = source
    cp, mom = parallel_transport(
        source, target, atlas, name, output_dir, registration_args, transport_args,
        shoot_args)

    # Compute the velocity field at atlas cp because all patients have momentas
    # supported at different control points, and velocity is more stable/smooth
    vel = kernel.convolve(atlas_cp, cp, mom)
    series = pd.Series([name] + vel.flatten().tolist())
    transport_result = transport_result.append(series, ignore_index=True)

transport_result.columns = ['pid'] + [
    f'vel_{k}_{i}' for k in range(n_cp) for i in range(1, 4)]
transport_result = transport_result.set_index('pid')
vel_cols = transport_result.columns[transport_result.columns.str.contains('vel')]

# Add the targetVariable column, depressed? Mother?
transport_result['targetVariable'] = 'control'  # 'test'
transport_result.to_csv(output_dir / 'transported_vel.csv')

grouped = transport_result.groupby('targetVariable').mean()
controls = transport_result.loc[transport_result.targetVariable == 'control', vel_cols]
patients = transport_result.loc[transport_result.targetVariable == 'PPD', vel_cols]

# Hotelling test + Bonferroni correction
pval, t2 = hotelling_t2(controls, patients)
reject, corrected_pval, _, fwer = sm.stats.multipletests(pval, method='bonferroni')

mean_vel_ppd = grouped.loc['test', vel_cols].values
mean_vel_ctl = grouped.loc['control', vel_cols].values
significative_vel = reject[:, None] * mean_vel_ppd
significative_vel_dif = reject[:, None] * (mean_vel_ppd - mean_vel_ctl).numpy()

# Add to a vtk file to visualize results
poly = pv.PolyData(atlas_cp)
poly['SignificantMeanVelocity'] = significative_vel
poly['SignificantMeanVelocityDiff'] = significative_vel_dif
poly['Hotelling_pval'] = np.log(corrected_pval)
poly['Hotelling_reject'] = reject.astype(float)
poly['MeanVelocityPPD'] = mean_vel_ppd
poly['MeanVelocityControl'] = mean_vel_ctl
poly.save(output_dir / 'hotelling_cp.vtk')

# CCA

# Make sure epds scores are numerical variables, otherwise use one hot encoding
# Make sure subjects follow the same order in both data frames.
epds_scores = pd.read_csv('')
cca = CCA(n_components=2)
X_train = transport_result[vel_cols]
Y_train = epds_scores
cca.fit(X_train, Y_train)
X_train_r, Y_train_r = cca.transform(X_train, Y_train)


# On diagonal plot X vs Y scores on each components
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.legend(loc="best")

plt.subplot(224)
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.legend(loc="best")

# Off diagonal plot components 1 vs 2 for X and Y
plt.subplot(222)
plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train", marker="*", s=50)
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.legend(loc="best")

plt.subplot(223)
plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train", marker="*", s=50)
plt.xlabel("Y comp. 1")
plt.ylabel("Y comp. 2")
plt.legend(loc="best")
plt.show()
