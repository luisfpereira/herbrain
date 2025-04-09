import dash_bootstrap_components as dbc
from dash import Dash, html
from polpo.dash.hydra import load_variables
from polpo.dash.style import update_style
from polpo.hydra import instantiate_dict_from_config, load_data, load_models


def my_app(cfg):
    style = cfg.style
    update_style(style)

    load_variables(cfg.vars, name="var")
    load_data(cfg.data, name="data")
    load_models(cfg.models, name="model")
    objs = instantiate_dict_from_config(cfg.objs, name="obj")

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        # suppress_callback_exceptions=True,
        # TODO: uncomment
        # assets_folder=cfg.app.assets_folder,
    )

    app.layout = dbc.Container(
        [
            html.H1("Brain Shape Prediction with Hormones, Menstrual"),
            html.Hr(),
        ]
        + objs["mesh_explorer"].to_dash(),
        fluid=True,
    )
    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )
