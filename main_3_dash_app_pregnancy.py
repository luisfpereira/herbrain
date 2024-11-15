"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import dash_bootstrap_components as dbc
import hydra
import numpy as np
from dash import Dash, Input, Output, callback, html
from dash_gi.style import update_style
from hydra.utils import instantiate
from omegaconf import OmegaConf

import project_pregnancy.app.page_content as page_content
from project_pregnancy.app.registry import PAGES

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

    # TODO: move this to dash-gi?
    OmegaConf.register_new_resolver("var", lambda key: variables[key])
    OmegaConf.register_new_resolver("data", lambda key: data[key])

    mri_explorer = instantiate(cfg.mri_explorer)
    mesh_explorer = instantiate(cfg.mesh_explorer)

    # create pages
    explore_data_page = page_content.explore_data(mri_explorer)
    ai_hormone_prediction_page = page_content.ai_hormone_prediction(mesh_explorer)
    home_page = page_content.homepage()

    app.layout = page_content.app_layout()
    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run_server(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )


if __name__ == "__main__":
    my_app()
