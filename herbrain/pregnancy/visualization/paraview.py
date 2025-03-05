import json
import re
from pathlib import Path

import herbrain.lddmm.strings as lddmm_strings


def generate_visualization(registration_dir, regression_dir, data_set, times):
    filenames = dict(
        registration={
            "registrationName": "registration",
            "FileNames": [
                (registration_dir / lddmm_strings.registration_str.format(i)).as_posix()
                for i in range(11)
            ],
        },
        initial_cp_registration={
            "registrationName": "initial_control_points_registration",
            "FileNames": [(registration_dir / "initial_control_points.vtk").as_posix()],
        },
        target_shape={
            "registrationName": "target_shape",
            "FileNames": [(registration_dir / "target_shape.vtk").as_posix()],
        },
        source_shape={
            "registrationName": "source",
            "FileNames": [(registration_dir / lddmm_strings.template_str).as_posix()],
        },
        initial_cp_regression={
            "registrationName": "initial_control_points_regression",
            "FileNames": [(regression_dir / "initial_control_points.vtk").as_posix()],
        },
        regression_shapes={
            "registrationName": "regression",
            "FileNames": [
                (regression_dir / lddmm_strings.regression_str.format(i, t)).as_posix()
                for i, t in enumerate(sorted(times))
            ],
        },
        raw_data_shapes={
            "registrationName": "raw_data",
            "FileNames": [k["shape"].as_posix() for k in data_set],
        },
    )

    with open(registration_dir.parent / f"visualisation_names.json", "w") as fp:
        json.dump(filenames, fp)

    template_path = Path(__file__).parent / "template.py"
    with open(template_path, "r") as file:
        content = file.read()

    pattern = re.compile("result_path_place_holder")
    updated_content = pattern.sub(registration_dir.parent.as_posix(), content)
    with open(registration_dir.parent / "viz.py", "w") as file:
        file.write(updated_content)


def update_visualization(project_dir, structure, config):
    dir_structure_config = project_dir / "meshes_nico" / structure / config
    with open(dir_structure_config / "visualisation_names.json", "r") as f:
        names = json.load(f)

    old_base = "/user/nguigui/home/Documents/UCSB/pregnancy"

    new_names = names.copy()
    for k, v in names.items():
        new_names[k]["FileNames"] = [
            path.replace(old_base, project_dir.as_posix())
            for path in names[k]["FileNames"]
        ]

    with open(dir_structure_config / "visualisation_names.json", "w") as f:
        json.dump(names, f)

    with open(dir_structure_config / "viz.py", "r") as file:
        content = file.read()

    pattern = re.compile(old_base)
    updated_content = pattern.sub(project_dir.as_posix(), content)
    with open(dir_structure_config / "viz.py", "w") as file:
        file.write(updated_content)
