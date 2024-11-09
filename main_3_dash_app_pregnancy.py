"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import os
import random

import dash_bootstrap_components as dbc
import hydra
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html
from hydra.utils import instantiate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs
import nibabel as nib
from dash_gi.components import ComponentGroup
from dash_gi.style import update_style
from omegaconf import OmegaConf

import project_pregnancy.app.calculations as calculations
import project_pregnancy.app.page_content as page_content
import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
from project_pregnancy.app.registry import PAGES
from src.preprocessing import smoothing
from src.regression import training

# TODO: replace print by logging


# (
#     space,
#     mesh_sequence_vertices,
#     vertex_colors,
#     hormones_df,
#     full_hormones_df,
# ) = data_utils.load_real_data(default_config, return_og_segmentation=False)
# # Do not include postpartum values that are too low
# hormones_df = hormones_df[hormones_df["EndoStatus"] == "Pregnant"]
# # convert sessionID sess-01 formatting to 1 for all entries
# # hormones_df['session_number'] = hormones_df['sessionID'].str.extract('(\d+)') #.astype(int)

# mesh_sequence_vertices = mesh_sequence_vertices[
#     :9
# ]  # HACKALART: first 9 meshes are pregnancy

# # Load MRI data
# raw_mri_dict = data_utils.load_raw_mri_data(default_config.raw_preg_mri_dir)

# X_hormones = hormones_df[["estro", "prog", "lh"]].values
# _, n_hormones = X_hormones.shape
# X_hormones_mean = X_hormones.mean(axis=0)

# (
#     lr_hormones,
#     pca_hormones,
#     y_mean_hormones,
#     n_vertices_hormones,
#     mesh_neighbors_hormones,
# ) = calculations.train_lr_model(X_hormones, mesh_sequence_vertices, n_hormones)


@callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    """Render the page content based on the URL."""

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


def linear_interpolation(x_lower, x_higher, y_lower, y_upper, x_input):
    """Linear interpolation between two points."""
    # Calculate the slope
    m = (y_upper - y_lower) / (x_higher - x_lower)

    # Calculate the interpolated y value
    y = y_lower + m * (x_input - x_lower)

    return y


def interpolate_or_return(df, x, x_label, y_label):
    """Interpolate or return the y value based on the x value."""
    print("interpolating")

    # Extract x and y values from dataframe
    x_values = df[x_label].values
    y_values = df[y_label].values

    # Check if x is within the range of known x values
    if x < x_values[0]:
        # Extrapolate using the first two data points
        x_lower = x_values[0]
        x_upper = x_values[1]
        y_lower = y_values[0]
        y_upper = y_values[1]

        # Perform linear extrapolation
        interpolated_y = linear_interpolation(x_lower, x_upper, y_lower, y_upper, x)

        return interpolated_y
    elif x in x_values:
        # If x is found, return the corresponding y value
        return y_values[np.where(x_values == x)[0][0]]
    else:
        # If x is not found, find the two nearest x values
        closest_idx = np.abs(x_values - x).argmin()
        if x_values[closest_idx] < x:
            lower_index = closest_idx
            upper_index = closest_idx + 1
        else:
            upper_index = closest_idx
            lower_index = closest_idx - 1

        print("index found")
        x_lower = x_values[lower_index]
        print("x_lower found")
        x_upper = x_values[upper_index]
        y_lower = y_values[lower_index]
        y_upper = y_values[upper_index]

        print(x_lower, x_upper, y_lower, y_upper, x)

        # Perform linear interpolation
        return linear_interpolation(x_lower, x_upper, y_lower, y_upper, x)


# @app.callback(
#     [
#         Output("mesh-plot", "figure"),
#         Output("gest_week_slider_container", component_property="style"),
#         Output("hormone_slider_container", component_property="style"),
#     ],
#     Input("gest-week-slider", "drag_value"),
#     Input("estrogen-slider", "drag_value"),
#     Input("progesterone-slider", "drag_value"),
#     Input("LH-slider", "drag_value"),
#     State("mesh-plot", "figure"),
#     State("mesh-plot", "relayoutData"),
#     Input("button", "n_clicks"),
# )
# def update_mesh(
#     gest_week, estrogen, progesterone, LH, current_figure, relayoutData, n_clicks=0
# ):
#     """Update the mesh plot based on the hormone levels."""
#     if (n_clicks % 2) == 0:
#         gest_week_slider_style = {"display": "none"}
#         hormone_week_slider_style = {"display": "block"}

#     else:
#         gest_week_slider_style = {"display": "block"}
#         hormone_week_slider_style = {"display": "none"}

#         print("hiding hormone sliders")

#         progesterone = interpolate_or_return(
#             hormones_df, gest_week, x_label="gestWeek", y_label="prog"
#         )
#         estrogen = interpolate_or_return(
#             hormones_df, gest_week, x_label="gestWeek", y_label="estro"
#         )
#         LH = interpolate_or_return(
#             hormones_df, gest_week, x_label="gestWeek", y_label="lh"
#         )
#         print("progesterone", progesterone)
#         print("estrogen", estrogen)
#         print("LH", LH)
#         print("gest_week", gest_week)

#     X_multiple = gs.array([[estrogen, progesterone, LH]])

#     mesh_plot = calculations.predict_mesh(
#         X_multiple,
#         lr_hormones,
#         pca_hormones,
#         y_mean_hormones,
#         n_vertices_hormones,
#         mesh_neighbors_hormones,
#         space,
#         vertex_colors,
#         current_figure=current_figure,
#         relayoutData=relayoutData,
#     )

#     print(n_clicks)

#     return mesh_plot, gest_week_slider_style, hormone_week_slider_style


@hydra.main(version_base=None, config_path="./config", config_name="config")
def my_app(cfg):
    style = cfg.style
    update_style(style)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=cfg.app.assets_folder,
    )

    variables = {
        key: instantiate(value, id_=key) for key, value in cfg.variables.items()
    }
    data = {key: instantiate(value).load() for key, value in cfg.data.items()}

    OmegaConf.register_new_resolver("var", lambda key: variables[key])
    OmegaConf.register_new_resolver("data", lambda key: data[key])

    mri_explorer = instantiate(cfg.mri_explorer)

    explore_data_page = page_content.explore_data(mri_explorer)

    hormone_sliders = ComponentGroup(
        [
            instantiate(value, var_def=variables[key])
            for key, value in cfg.hormone_sliders.items()
        ]
    )
    gest_week_sliders = ComponentGroup(
        [
            instantiate(value, var_def=variables[key])
            for key, value in cfg.gest_week_sliders.items()
        ]
    )

    ai_hormone_prediction_page = page_content.ai_hormone_prediction(
        gest_week_sliders, hormone_sliders
    )

    sidebar = page_content.sidebar()
    home_page = page_content.homepage()

    # TODO: move to differnet place?
    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            sidebar,
            content,
        ]
    )

    # create callbacks
    # TODO: update here
    # create_update_nii_plot_basic(raw_mri_data)

    server_cfg = cfg.server
    app.run_server(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )


if __name__ == "__main__":
    my_app()
