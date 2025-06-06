import pandas as pd
import pyvista as pv


def preprocess_full_structure(mesh: pv.DataSet, target_mesh=None):
    """Align, smooth and decimate mesh."""
    aligned = mesh.align(target_mesh, max_iterations=100)
    new_mesh = aligned.smooth_taubin(n_iter=20)
    target_reduction = 1 - (5000 / mesh.n_points)
    new_mesh = new_mesh.decimate(target_reduction)
    return new_mesh, aligned


def add_labels(mesh: pv.DataSet, labels: dict):
    """Add names of structure on vertices of the mesh."""
    new_mesh = mesh
    labels_df = pd.DataFrame(new_mesh.get_array("RGBA"))
    labels_df["label"] = labels_df.apply(tuple, axis=1).map(labels)
    new_mesh["label"] = labels_df["label"]
    return new_mesh


def preprocess_substructure(
    mesh: pv.DataSet, label: str, labels: dict, align=False, target_mesh=None
):
    """Align, smooth and decimate mesh of substructure."""
    new_mesh = mesh
    labels_df = pd.DataFrame(new_mesh.get_array("RGBA"))
    labels_df["label"] = labels_df.apply(tuple, axis=1).map(labels)
    new_mesh["label"] = labels_df["label"]
    structure = new_mesh.extract_points(
        labels_df.loc[labels_df.label == label].index, adjacent_cells=False
    )
    if align and target_mesh is not None:
        structure = structure.align(target_mesh, max_iterations=10)
    smoothed = structure.extract_surface().smooth_taubin(20)
    if smoothed.n_points > 2000:
        return smoothed.decimate(0.2)
    return smoothed


def get_ref_labels():
    return {
        (255, 0, 255, 255): "PRC",
        (0, 255, 255, 255): "PHC",
        (255, 215, 0, 255): "AntHipp",
        (255, 255, 0, 255): "ERC",
        (80, 179, 221, 255): "SUB",
        (184, 115, 51, 255): "PostHipp",
        (255, 0, 0, 255): "CA1",
        (0, 0, 255, 255): "DG",
        (0, 255, 0, 255): "CA2+3",
    }


def swap_left_right(days, data_dir):
    """Swap left / right meshes for given days in data_dir."""
    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    for day in days:
        file_left = data_dir / f"left_structure_-1_day{day:02}.ply"
        file_left_tmp = tmp_dir / file_left.name
        file_left.rename(file_left_tmp)
        file_right = data_dir / f"right_structure_-1_day{day:02}.ply"
        file_right.rename(data_dir / f"left_structure_-1_day{day:02}.ply")
        file_left_tmp.rename(file_right)


def main(day_min, day_max, day_ref, side, data_dir, output_dir):
    """Loop preprocessing over observations between day_min and day_max.

    For each session and a given side (left /right) :
        - Align, smooth and decimate the whole mesh;
        - extract substructures;
        - align, smooth and decimate each substructure.

    Save the results in output_dir, by structure.
    """
    output_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "full_structure" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    smooth_dir = output_dir / "full_structure" / "smooth"
    smooth_dir.mkdir(parents=True, exist_ok=True)

    ref_labels = get_ref_labels()
    zones = ref_labels.values()

    for z in zones:
        (output_dir / z / "raw").mkdir(parents=True, exist_ok=True)

    ref_mesh = pv.read(data_dir / f"{side}_structure_-1_day{day_ref:02}.ply")
    target_struct = {k: None for k in zones}

    for d in range(day_min, day_max + 1):
        name = f"{side}_structure_-1_day{d:02}.ply"
        decimate_name = f"{side}_full_{d:02}.vtk"

        try:
            # center, smooth and reduce nb of vertices of full structure
            day_mesh = pv.read(data_dir / name)
            day_mesh = add_labels(day_mesh, ref_labels)
            day_mesh.save(raw_dir / decimate_name)
            preproc, aligned_day = preprocess_full_structure(day_mesh, ref_mesh)
            preproc.save(smooth_dir / decimate_name)

            # extract sub regeions, center and smooth each
            for z in zones:
                struc_name = f"{side}_{z}_t{d:02}.vtk"
                substruc_mesh = preprocess_substructure(
                    aligned_day,
                    z,
                    ref_labels,
                    align=(d > 1),
                    target_mesh=target_struct[z],
                )
                substruc_mesh.save(output_dir / z / "raw" / struc_name)
                if d == 1:
                    target_struct[z] = substruc_mesh

        except FileNotFoundError:
            continue
