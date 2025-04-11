import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from hydra import compose, initialize
from polpo.hydra import DefaultsResolver

app = typer.Typer()


def _launch_app(my_app, overrides, config_path, config_name, logging_level):
    logging.basicConfig(level=logging_level)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    my_app(cfg)


@app.command()
def pregnancy_app(
    overrides: Optional[List[str]] = typer.Argument(None),
    config_path: str = "config/pregnancy",
    config_name: str = "config",
    logging_level: int = 20,
):
    """Launch pregnancy app."""
    from herbrain.pregnancy.app import my_app

    if overrides is None:
        overrides = []

    overrides = DefaultsResolver().resolve(
        Path(os.path.dirname(__file__)) / config_path,
        key="data",
        overrides=overrides,
        config_name=config_name,
    )

    return _launch_app(my_app, overrides, config_path, config_name, logging_level)


@app.command()
def menstrual_app(
    overrides: Optional[List[str]] = typer.Argument(None),
    config_path: str = "config/menstrual",
    config_name: str = "config",
    logging_level: int = 20,
):
    """Launch menstrual app."""
    from herbrain.menstrual.app import my_app

    return _launch_app(my_app, overrides, config_path, config_name, logging_level)


if __name__ == "__main__":
    app()
