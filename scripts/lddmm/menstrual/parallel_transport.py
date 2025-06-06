import subprocess
from pathlib import Path

import pandas as pd
import preprocessing

import herbrain.lddmm as lddmm
import herbrain.lddmm.strings as lddmm_strings
from herbrain.lddmm.regression import main as regression

project_dir = Path("/user/nguigui/home/Documents/UCSB/menstrual")
data_dir = project_dir / "a_meshed"
output_dir = project_dir / "meshes_nico"
tmp_dir = Path("/tmp")
ref_labels = preprocessing.get_ref_labels()


def get_day(side, struct, day):
    name = f"{side}_{struct}_t{day:02}.vtk"
    return output_dir / structure / "raw" / name


side_ = "left"
structure = "PostHipp"
# preprocessing.main(1, 60, 1, side_, data_dir, output_dir)

# registration of day 1 - main geodesic
atlas = get_day(side_, structure, day=1)
target = get_day(side_, structure, day=31)

registration_dir = output_dir / structure / "initial_registration"
registration_args = dict(
    kernel_width=6.0,
    regularisation=1.0,
    max_iter=2000,
    freeze_control_points=False,
    attachment_kernel_width=2.0,
    metric="varifold",
    tol=1e-10,
    filter_cp=True,
    threshold=0.75,
    use_rk4_for_shoot=True,
    use_rk2_for_shoot=False,
)

# lddmm.registration(atlas, target, registration_dir, **registration_args)

# registration, transport, and shoot each day
registration_args.update(use_rk4_for_shoot=False, use_rk2_for_shoot=True)
registration_seq_dir = output_dir / structure / "seq_registration"
registration_seq_dir.mkdir(exist_ok=True)

source = get_day(side_, structure, day=31)
n_rungs = 11
transport_args = {"kernel_type": "torch", "kernel_width": 4, "n_rungs": n_rungs}

shoot_args = {
    "source": source,
    "use_rk2_for_flow": True,
    "kernel_width": 4.0,
    "number_of_time_steps": n_rungs + 1,
    "write_params": False,
}

momenta = registration_dir / lddmm_strings.momenta_str
control_points = registration_dir / lddmm_strings.cp_str

for d in range(31, 61):
    target = get_day(side_, structure, day=d)
    output_name = tmp_dir / structure / f"reg_{d}"
    lddmm.registration(source, target, output_name, **registration_args)

    to_move = [
        lddmm_strings.cp_str,
        lddmm_strings.momenta_str,
        lddmm_strings.residual_str,
    ]
    move_to = [f"cp_{d}.txt", f"momenta_{d}.txt", f"registration_error_{d}.txt"]

    for s, t in zip(to_move, move_to):
        subprocess.call(["mv", output_name / s, registration_seq_dir / t])

    output_name = tmp_dir / f"{d}_transport_S"
    subprocess.call(["mkdir", output_name])

    target_dir = output_dir / structure / "transport"
    momenta_to_transport = (registration_seq_dir / f"momenta_{d}.txt").as_posstringsix()
    control_points_to_transport = (registration_seq_dir / f"cp_{d}.txt").as_posix()

    subprocess.call(["mkdir", target_dir])

    lddmm.transport(
        control_points,
        momenta,
        control_points_to_transport,
        momenta_to_transport,
        output_name,
        **transport_args,
    )

    to_move = ["final_cp.txt", "transported_momenta.txt"]
    transported_cp = target_dir / f"transported_cp_{d}.txt"
    transported_mom = target_dir / f"transported_momenta_{d}.txt"
    move_to = [transported_cp, transported_mom]

    for src, dest in zip(to_move, move_to):
        subprocess.call(["mv", output_name / src, dest])

    # Shoot transported momenta from atlas
    lddmm.shoot(
        control_points=transported_cp.as_posix(),
        momenta=transported_mom.as_posix(),
        output_dir=output_name,
        **shoot_args,
    )

    shoot_name = output_name / lddmm_strings.shoot_str.format(n_rungs)
    subprocess.call(["mv", shoot_name, target_dir / f"transported_shoot_{d}.vtk"])
    subprocess.call(["rm", "-r", output_name])


covariates = pd.read_csv(project_dir / "hormones.csv", index_col=0)
covariates.loc[covariates.index <= 30, "cycle"] = "free"
covariates.loc[covariates.index > 30, "cycle"] = "birthcontrol"

nice_zones = ["PRC", "PHC", "PostHipp", "CA2+3", "ERC"]
covariates[nice_zones] = True
covariates.loc[covariates.index == 56, nice_zones] = False
# fig, ax = plt.subplots(figsize=(8, 6))
# hormones.groupby('cycle').plot(x='CycleDay', y='Prog', ax=ax)
# plt.show()

registration_args = dict(
    kernel_width=6.0,
    regularisation=1.0,
    max_iter=2000,
    freeze_control_points=False,
    metric="varifold",
    attachment_kernel_width=1.0,
    tol=1e-10,
    filter_cp=True,
    threshold=0.75,
)
spline_args = dict(
    initial_step_size=100,
    regularisation=1.0,
    freeze_external_forces=True,
    freeze_control_points=True,
)

variable = "CycleDay"
data_list = covariates[covariates[structure]]  # remove excluded
data_list = data_list[data_list.cycle == "birthcontrol"]
data_list = data_list.dropna(subset=variable)
data_list = data_list.sort_values(by=variable)
times = data_list[variable]
times = (times - times.min()) / (times.max() - times.min())

data_path = output_dir / structure / "transport"
dataset = [{"shape": data_path / f"transported_shoot_{k}.vtk"} for k in data_list.index]
regression(
    dataset, times, output_dir, "time_bc", structure, registration_args, spline_args
)


data_list = covariates[covariates[structure]]  # remove excluded
data_list = data_list[data_list.cycle == "free"]
data_list = data_list.dropna(subset=variable)
data_list = data_list.sort_values(by="CycleDay")
times = data_list.CycleDay
times = (times - times.min()) / (times.max() - times.min())

data_path = output_dir / structure / "raw"
dataset = [
    {"shape": data_path / f"{side_}_{structure}_t{k:02}.vtk"} for k in data_list.index
]
regression(
    dataset, times, output_dir, "time_free", structure, registration_args, spline_args
)
