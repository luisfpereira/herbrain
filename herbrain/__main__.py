import logging
from typing import List, Optional

import typer
from hydra import compose, initialize

app = typer.Typer()


@app.command()
def pregnancy_app(
    overrides: Optional[List[str]] = (),
    config_path: str = "pregnancy/config",
    config_name: str = "config",
    logging_level: int = 20,
):
    """Launch pregnancy app."""
    from herbrain.pregnancy.app import my_app

    logging.basicConfig(level=logging_level)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    my_app(cfg)


if __name__ == "__main__":
    app()
