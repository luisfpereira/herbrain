import dash_bootstrap_components as dbc
from dash import Dash, html
from hydra.utils import instantiate
from polpo.dash.hydra import load_variables
from polpo.dash.style import update_style
from polpo.hydra import load_data, load_models


def my_app(cfg):
    style = cfg.style
    update_style(style)

    load_variables(cfg.variables, name="var")
    load_data(cfg.data, name="data")
    load_models(cfg.models, name="model")

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        # suppress_callback_exceptions=True,
        # TODO: uncomment
        # assets_folder=cfg.app.assets_folder,
    )

    mesh_explorer = instantiate(cfg.mesh_explorer)

    app.layout = dbc.Container(
        [
            html.H1("Brain Shape Prediction with Hormones, Menstrual"),
            html.Hr(),
        ]
        + mesh_explorer.to_dash(),
        fluid=True,
    )
    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run_server(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )
