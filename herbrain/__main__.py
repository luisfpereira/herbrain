import typer

app = typer.Typer()


@app.command()
def pregnancy_app():
    """Launch pregnancy app."""
    from herbrain.pregnancy.app import my_app

    my_app()


if __name__ == "__main__":
    app()
