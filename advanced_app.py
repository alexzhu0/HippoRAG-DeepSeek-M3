"""Advanced CLI with history and CSV export; HippoRAGDemo remains importable."""

from cli import run_cli
from demo_core import HippoRAGDemo


def main(argv=None):
    return run_cli(HippoRAGDemo, advanced=True, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
