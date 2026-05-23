"""CLI entry point: paper-distiller-web.

Usage:
    paper-distiller-web --vault /path/to/vault [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="paper-distiller-web",
        description="Launch the paper-distiller web frontend.",
    )
    parser.add_argument(
        "--vault",
        required=True,
        metavar="PATH",
        help="Path to the vault directory.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Bind port (default: 8765).",
    )
    args = parser.parse_args()

    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "uvicorn is not installed. Install with:\n"
            '  pip install "paper-distiller[web]"',
            file=sys.stderr,
        )
        sys.exit(1)

    from .server import create_app  # noqa: PLC0415

    app = create_app(args.vault)
    print(f"Starting paper-distiller web on http://{args.host}:{args.port}")
    print(f"Vault: {args.vault}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
