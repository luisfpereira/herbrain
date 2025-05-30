import numpy as np
import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import (
    IfCondition,
    IndexMap,
    ListSqueeze,
    Map,
    NestingSwapper,
    PartiallyInitializedStep,
    Pipeline,
    Sorter,
    Truncater,
)
from polpo.preprocessing.load.pregnancy import (
    DenseMaternalCsvDataLoader,
    DenseMaternalMeshLoader,
    PregnancyPilotMriLoader,
    PregnancyPilotRegisteredMeshesLoader,
    PregnancyPilotSegmentationsLoader,
)
from polpo.preprocessing.mesh.conversion import TrimeshFromData, TrimeshFromPv
from polpo.preprocessing.mesh.io import PvReader, TrimeshReader
from polpo.preprocessing.mesh.registration import PvAlign
from polpo.preprocessing.mri import (
    MriImageLoader,
    SkimageMarchingCubes,
)

# TODO: check docstrings

# TODO: Pipeline -> Pipe

# TODO: if general enough, move to polpo


class PilotMriImageLoader(Pipeline):
    """Load, sort, truncate, and parse MRI images."""

    def __init__(self, debug=False):
        if debug:
            value = 2
            n_jobs = 1
            verbose = 1
        else:
            value = None
            n_jobs = -1
            verbose = 0

        super().__init__(
            steps=[
                PregnancyPilotMriLoader(as_dict=False),
                Sorter(),
                Truncater(value=value),
                Map(n_jobs=n_jobs, verbose=verbose, step=MriImageLoader()),
            ]
        )


class HormonesCsvLoader(Pipeline):
    """Load maternal hormone data and drop repeated row."""

    def __init__(self):
        super().__init__(
            steps=[
                DenseMaternalCsvDataLoader(pilot=True),
                ppd.Drop(labels=27),
            ]
        )


class TemplateImageLoader(Pipeline):
    """Load and parse a single MRI image as a template with affine."""

    def __init__(self):
        super().__init__(
            steps=[
                PregnancyPilotMriLoader(subset=[1]),
                ListSqueeze(),
                MriImageLoader(as_nib=True),
            ]
        )


class ReferenceImageLoader(Pipeline):
    """Load and parse a single MRI image as a template with affine."""

    def __init__(self):
        super().__init__(
            steps=[
                PregnancyPilotSegmentationsLoader(subset=[1]),
                ListSqueeze(),
                MriImageLoader(as_nib=True),
            ]
        )


class NibImage2Mesh(Pipeline):
    """Generate a surface mesh from a 3D MRI image."""

    def __init__(self):
        super().__init__(
            steps=[
                lambda x: x.get_fdata(),
                SkimageMarchingCubes(return_values=False),
                TrimeshFromData(),
            ]
        )


class HippRegisteredMeshesLoader(Pipeline):
    """Load and parse registered meshes per session."""

    def __init__(self):
        super().__init__(
            steps=[
                PregnancyPilotRegisteredMeshesLoader(
                    method="deformetrica", as_dict=True
                ),
                ppdict.DictMap(step=TrimeshReader()),
            ]
        )


class HippReferenceImagePipe(Pipeline):
    """Load a single segmentation MRI image with its affine."""

    def __init__(self):
        super().__init__(
            steps=[
                PregnancyPilotSegmentationsLoader(subset=[1]),
                ListSqueeze(),
                PilotMriImageLoader(return_affine=True),
            ]
        )


class MaternalRegisteredMeshesLoader(Pipeline):
    """
    Load, align, and convert a set of per-subject meshes to trimesh format.

    The pipeline loads meshes in Pv format, aligns them to a template,
    and converts them into `trimesh.Trimesh` objects. The key is the subject/session ID.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of iterations for alignment.
    """

    def __init__(self, max_iterations=500):
        # TODO: allow template choice?
        super().__init__(
            steps=[
                DenseMaternalMeshLoader(subject_id=None, as_dict=True),
                ppdict.DictMap(step=PvReader()),
                PartiallyInitializedStep(
                    Step=lambda target: ppdict.DictMap(
                        PvAlign(target=target, max_iterations=max_iterations)
                    ),
                    _target=lambda meshes: meshes[1],  # select template
                ),
                ppdict.DictMap(step=TrimeshFromPv()),
            ]
        )


class MultipleMaternalMeshesLoader(Pipeline):
    """
    Load, align, and convert Pv meshes for a list of brain structures.

    This pipeline uses structure names to determine lateralization,
    loads meshes using `DenseMaternalMeshLoader`, parses Pv files, and applies
    joint registration + conversion to Trimesh format.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum iterations for PvAlign. Default is 500.
    """

    # TODO: fix docstrings

    def __init__(self, max_iterations=500):
        super().__init__(
            steps=[
                ppdict.HashWithIncoming(
                    step=Map(
                        step=Pipeline(
                            steps=[
                                PartiallyInitializedStep(
                                    Step=DenseMaternalMeshLoader,
                                    pass_data=False,
                                    _struct=lambda name: name.split("_")[-1],
                                    _left=lambda name: name.split("_")[0] == "L",
                                    as_dict=True,
                                ),
                                ppdict.DictMap(step=PvReader()),
                            ]
                        )
                    )
                ),
                ppdict.DictMap(
                    step=PartiallyInitializedStep(
                        Step=lambda target, max_iterations: ppdict.DictMap(
                            PvAlign(target=target, max_iterations=max_iterations)
                            + TrimeshFromPv()
                        ),
                        _target=lambda meshes: meshes[list(meshes.keys())[0]],
                        max_iterations=max_iterations,
                    )
                ),
            ]
        )


class SessionInputMultipleMeshToXY(Pipeline):
    # TODO: improve naming?
    # TODO: equivalent to DictsToXY: generalize and move to polpo?
    def __init__(self):
        x_step = IfCondition(
            step=Map(step=ppdict.DictToValuesList()),
            else_step=lambda x: np.asarray(x)[:, None],
            condition=lambda x: isinstance(x[0], dict),
        )

        super().__init__(
            steps=[
                IndexMap(index=1, step=ppdict.NestedDictSwapper()),
                ppdict.DictMerger(),
                NestingSwapper(),
                IndexMap(index=0, step=x_step),
                IndexMap(index=1, step=ppdict.ListDictSwapper()),
            ]
        )
