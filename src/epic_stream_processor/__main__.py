"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Epic Stream Processor."""


if __name__ == "__main__":
    main(prog_name="epic-stream-processor")  # pragma: no cover
