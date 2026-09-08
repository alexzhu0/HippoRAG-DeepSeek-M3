"""Basic CLI. Importing this module never initializes models or credentials."""

from cli import run_cli
from demo_core import HippoRAGDemo


def main(argv=None):
    return run_cli(HippoRAGDemo, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
