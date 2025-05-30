"""Creates a Dash app where week/hormone sliders predict brain shape."""

import dash_bootstrap_components as dbc
import numpy as np
import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from dash import Dash
from polpo.dash.callbacks import PageRegister
from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    FunctionComponent,
    Graph,
    MriExplorer,
    MriSliders,
    MultiModelsMeshExplorer,
    SidebarElem,
    SidebarHeader,
    Slider,
)
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import DictMeshColorizer, MeshColorizer
from polpo.plot.mesh import MeshesPlotter, MeshPlotter, StaticMeshPlotter
from polpo.preprocessing import ListSqueeze
from polpo.preprocessing.learning import DictsToXY, NestedDictsToXY
from polpo.preprocessing.mri import (
    LocalToTemplateTransform,
)
from polpo.sklearn.compose import PostTransformingEstimator

import herbrain.pregnancy.page_content as page_content

from .data import (
    HippRegisteredMeshesLoader,
    HormonesCsvLoader,
    MaternalRegisteredMeshesLoader,
    MultipleMaternalMeshesLoader,
    NibImage2Mesh,
    PilotMriImageLoader,
    ReferenceImageLoader,
    TemplateImageLoader,
)
from .models import MeshPCR
from .page_content import ai_hormone_prediction, homepage, mri_page


def my_app(cfg, data="hipp"):
    style = cfg.style
    update_style(style)

    hormones_ordering = ["estro", "prog", "lh"]

    mri_data = PilotMriImageLoader(debug=cfg.server.debug)()
    hormones_df = HormonesCsvLoader()()

    hormones_for_pred = ppd.ColumnsToDict(hormones_ordering)(hormones_df)
    hormones_gest_week = ppd.ColumnToDict("gestWeek")(hormones_df)

    if data == "multiple":
        dicts_to_xy = NestedDictsToXY()
    else:
        dicts_to_xy = DictsToXY()

    template_image = TemplateImageLoader()()

    n_structs = 1
    if data == "hipp":
        registered_meshes = HippRegisteredMeshesLoader()()

        reference_image = ReferenceImageLoader()()
        affine_transform = LocalToTemplateTransform()(
            (reference_image.affine, template_image.affine)
        )

    elif data == "multiple":
        structs = [
            "BrStem",
            "L_Thal",
            "R_Thal",
            "L_Caud",
            "R_Caud",
            "L_Puta",
            "R_Puta",
            "L_Pall",
            "R_Pall",
            "L_Hipp",
            "R_Hipp",
            "L_Amyg",
            "R_Amyg",
            "L_Accu",
            "R_Accu",
        ]
        if cfg.server.debug:
            structs = structs[2:]

        n_structs = len(structs)
        registered_meshes = MultipleMaternalMeshesLoader(max_iterations=500)(structs)
        affine_transform = np.array(
            [
                [1.0, 0.0, 0.0, 25.0],
                [0.0, 1.0, 0.0, 28.0],
                [0.0, 0.0, 1.0, 23.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    else:
        registered_meshes = MaternalRegisteredMeshesLoader(max_iterations=500)()
        affine_transform = np.array(
            [
                [1.0, 0.0, 0.0, 25.0],
                [0.0, 1.0, 0.0, 28.0],
                [0.0, 0.0, 1.0, 23.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    n_pipes = n_structs if data == "multiple" else None
    week_mesh_model = MeshPCR(
        model=None, affine_transform=affine_transform, n_pipes=n_pipes
    )
    # TODO: scale input?
    hormones_mesh_model = MeshPCR(
        model=None, affine_transform=affine_transform, n_pipes=n_pipes
    )

    if data in ("maternal", "multiple"):
        Colorizer = MeshColorizer if data == "maternal" else DictMeshColorizer

        week_colorizer = Colorizer(x_ref=np.asarray(0.5), delta_lim=np.asarray(15.0))
        week_mesh_model = PostTransformingEstimator(week_mesh_model, week_colorizer)

        hormones_colorizer = Colorizer(scaling_factor=50.0)
        hormones_mesh_model = PostTransformingEstimator(
            hormones_mesh_model, hormones_colorizer
        )

    X, y = dicts_to_xy([hormones_gest_week, registered_meshes])
    week_mesh_model.fit(X, y)

    X, y = dicts_to_xy([hormones_for_pred, registered_meshes])
    hormones_mesh_model.fit(X, y)

    session_id = VarDef("sessionID", name="Session Number", min_value=1, max_value=26)
    mri_vars = [session_id] + [
        VarDef(id_, name=name)
        for id_, name in [
            ("mri_x", "X Coordinate (Changes Side View)"),
            ("mri_y", "Y Coordinate (Changes Front View)"),
            ("mri_z", "Z Coordinate (Changes Top View)"),
        ]
    ]
    gest_week = VarDef(
        "gestWeek", name="Gestational Week", min_value=0, max_value=36, default_value=15
    )
    estro = VarDef(
        "estro",
        name="Estrogen",
        unit="pg/ml",
        min_value=4100,
        max_value=12400,
    )
    prog = VarDef(
        "prog",
        name="Progesterone",
        unit="ng/ml",
        min_value=54,
        max_value=103,
    )
    lh = VarDef(
        "lh",
        name="LH",
        unit="ng/ml",
        min_value=0.59,
        max_value=1.45,
    )
    endo_status = VarDef("EndoStatus", name="Pregnancy status")
    trimester = VarDef("trimester", name="trimester")

    mri_steps = [1] + [5] * 3
    mri_sliders = MriSliders(
        [Slider(var, step) for var, step in zip(mri_vars, mri_steps)],
        trims=((20, 40), 50, 70),
    )

    session_info = ComponentGroup(
        components=[
            DepVar(var)
            for var in (session_id, gest_week, estro, lh, endo_status, trimester)
        ],
        title="Session information",
    )
    mri_explorer = MriExplorer(
        mri_data, hormones_df, mri_sliders, session_info, id_prefix="mri-"
    )

    template_mesh = NibImage2Mesh()(template_image)

    postproc_pred = None
    if data == "multiple":
        postproc_pred = ppdict.DictMap(step=ListSqueeze()) + ppdict.DictToValuesList()

    hormone_label_style = {"fontSize": 30, "display": "block"}
    mesh_explorer = MultiModelsMeshExplorer(
        graph=Graph(
            id_="mesh-plot",
            plotter=MeshesPlotter(
                plotters=[MeshPlotter() for _ in range(n_structs)],
                overlay_plotter=StaticMeshPlotter(mesh=template_mesh),
                bounds=None,  # TODO: check need
                overlay_bounds=None,  # TODO: check need
            ),
        ),
        models=(week_mesh_model, hormones_mesh_model),
        inputs=(
            Slider(gest_week),
            ComponentGroup(
                ordering=hormones_ordering,
                components=[
                    Slider(var, step, label_style=hormone_label_style)
                    for var, step in [(estro, 500), (prog, 3), (lh, 0.05)]
                ],
            ),
        ),
        checkbox_labels=((-1, "Show Full Brain", False),),
        button_label=" Click Here to Toggle Between Gestational Week vs Hormone Value Prediction",
        postproc_pred=postproc_pred,
    )

    sidebar_elems = [
        # home
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/", text="Home", image_url="home_emoji.jpeg"
            ),
            page=FunctionComponent(homepage),
        ),
        # mri explorer
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/page-1",
                text="Explore MRI Data",
                image_url="brain_emoji.jpeg",
                image_width=40,
            ),
            page=FunctionComponent(mri_page, mri_explorer=mri_explorer),
        ),
        # mesh explorer
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/page-2",
                text="AI: Hormones to Hippocampus Shape",
                image_url="robot_emoji.jpeg",
                image_width=40,
            ),
            page=FunctionComponent(
                ai_hormone_prediction,
                mesh_explorer=mesh_explorer,
                show_legend=True if data == "hipp" else False,
            ),
        ),
    ]

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=cfg.app.assets_folder,
    )

    page_register = PageRegister()

    app.layout = page_content.app_layout(sidebar_elems, page_register)
    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )
