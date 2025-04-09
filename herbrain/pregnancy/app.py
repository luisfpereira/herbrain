"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import dash_bootstrap_components as dbc
from dash import Dash
from polpo.dash.callbacks import PageRegister
from polpo.dash.hydra import load_variables
from polpo.dash.style import update_style
from polpo.hydra import instantiate_dict_from_config, load_data, load_models

import herbrain.pregnancy.page_content as page_content


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
        suppress_callback_exceptions=True,
        assets_folder=cfg.app.assets_folder,
    )

    page_register = PageRegister()

    app.layout = page_content.app_layout(objs["sidebar_elems"], page_register)
    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )
