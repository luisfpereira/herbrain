from polpo.models import DictMeshes2Comps, Meshes2Comps, ObjectRegressor
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.transform import AffineTransformation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def MeshPCR(model=None, affine_transform=None, n_components=4, n_pipes=None):
    """Linear regression on PCA components of transformed meshes."""
    # n_pipes: if dict with multiple structures

    if model is None:
        model = LinearRegression()

    mesh_transform = (
        Map(step=[AffineTransformation(transform=affine_transform)])
        if affine_transform is not None
        else None
    )

    if n_pipes is None:
        objs2y = Meshes2Comps(
            dim_reduction=PCA(n_components=n_components),
            smoother=False,
            mesh_transform=mesh_transform,
        )
    else:
        objs2y = DictMeshes2Comps(
            dim_reduction=PCA(n_components=n_components),
            smoother=False,
            mesh_transform=mesh_transform,
            n_pipes=n_pipes,
        )

    return ObjectRegressor(model=model, objs2y=objs2y)
