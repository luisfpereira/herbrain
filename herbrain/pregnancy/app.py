"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, html
from hydra.utils import instantiate
from polpo.dash.hydra import load_variables
from polpo.dash.style import update_style
from polpo.hydra import load_data, load_models

import herbrain.pregnancy.page_content as page_content
from herbrain.pregnancy.registry import PAGES


@callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    """Render the page content based on the URL."""
    # TODO: move to dash-gi?

    page = PAGES.get(pathname)
    if page is not None:
        return page

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


def my_app(cfg):
    style = cfg.style
    update_style(style)

    load_variables(cfg.variables, name="var")
    load_data(cfg.data, name="data")
    load_models(cfg.models, name="model")

    mri_explorer = instantiate(cfg.mri_explorer)
    mesh_explorer = instantiate(cfg.mesh_explorer, _convert_="object")

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=cfg.app.assets_folder,
    )

    # create pages
    page_content.explore_data(mri_explorer)
    page_content.ai_hormone_prediction(mesh_explorer)
    page_content.homepage()

    app.layout = page_content.app_layout()
    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run_server(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )
