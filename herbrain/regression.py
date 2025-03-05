import numpy as np
from pregnancy.visualization.paraview import generate_visualization

import herbrain.lddmm as lddmm
import herbrain.lddmm_strings as lddmm_strings


def main(
    data_set,
    times,
    output_dir,
    config_id="0",
    structure="PostHipp",
    registration_args=None,
    spline_args=None,
):
    """Geodesic regression.

    Perform a registration first to estimate control points, then a spline or geodesic
    regression. See the lddmm module for arguments of the registration and spline regression.
    """
    if registration_args is None:
        registration_args = {}
    if spline_args is None:
        spline_args = {}

    times = (times - times.min()) / (times.max() - times.min())
    target_weights = [1 / len(data_set)] * len(data_set)

    # registration to optimize control points
    source = data_set[0]["shape"]
    target = data_set[-1]["shape"]
    registration_dir = output_dir / structure / config_id / "initial_registration"
    lddmm.registration(source, target, registration_dir, **registration_args)

    # geodesic or spline regression (depending on freeze_external_forces)
    all_spline_args = registration_args.copy()
    all_spline_args.update(spline_args)
    all_spline_args["initial_control_points"] = registration_dir / lddmm_strings.cp_str
    regression_dir = output_dir / structure / config_id / "regression"
    lddmm.spline_regression(
        source=data_set[0]["shape"],
        targets=data_set,
        output_dir=regression_dir,
        times=times.tolist(),
        subject_id=[""],
        t0=min(times),
        target_weights=target_weights,
        **all_spline_args,
    )
    generate_visualization(registration_dir, regression_dir, data_set, times)


def compute_r2(data_set, out_dir, structure, config_id, registration_args):
    atlas_dir = out_dir / structure / config_id / "atlas"
    lddmm.deterministic_atlas(
        data_set[0]["shape"],
        data_set,
        structure,
        atlas_dir,
        initial_step_size=1e-1,
        **registration_args,
    )

    # Compute varifold distance between atlas and all datapoints
    atlas_dir_ = out_dir / structure / config_id / "atlas_frozen"
    del registration_args["max_iter"]
    lddmm.deterministic_atlas(
        atlas_dir / lddmm_strings.template_str,
        data_set,
        structure,
        atlas_dir_,
        initial_step_size=1e-10,
        max_iter=0,
        **registration_args,
    )

    regression_dir = out_dir / structure / config_id / "regression"
    residues_reg = lddmm.read_2D_array(
        regression_dir / lddmm_strings.residual_str_spline
    )
    residues_atlas = lddmm.read_2D_array(atlas_dir_ / lddmm_strings.residual_str)

    r2 = 1 - np.sum(residues_reg) / np.sum(residues_atlas)
    return r2
