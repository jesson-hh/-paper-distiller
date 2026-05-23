"""FastAPI application factory for paper-distiller web frontend.

Usage:
    from paper_distiller.web.server import create_app
    app = create_app("/path/to/vault")

The app:
- Serves static files from web/static/ at /
- Mounts REST + SSE API routes under /chat, /vault, /config, /healthz
- Injects vault_path into the HTML via a <meta> tag
- Does NOT store session state (client keeps full history)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def _inject_vault_meta(html: str, vault_path: str) -> str:
    """Insert <meta name="vault-path" content="..."> and rewrite static refs.

    The static files are served at /static/; rewrite relative refs so the
    browser finds them when the HTML is served at /.
    """
    tag = f'<meta name="vault-path" content="{vault_path}">'
    if "<head>" in html:
        html = html.replace("<head>", f"<head>\n  {tag}", 1)
    else:
        html = tag + "\n" + html

    # Rewrite relative static file references to /static/ prefix
    html = html.replace('href="paper-distiller.css"', 'href="/static/paper-distiller.css"')
    html = html.replace("href='paper-distiller.css'", "href='/static/paper-distiller.css'")
    html = html.replace('src="paper-distiller.jsx"', 'src="/static/paper-distiller.jsx"')
    html = html.replace("src='paper-distiller.jsx'", "src='/static/paper-distiller.jsx'")
    html = html.replace('src="deck-stage.js"', 'src="/static/deck-stage.js"')
    html = html.replace("src='deck-stage.js'", "src='/static/deck-stage.js'")
    return html


def create_app(vault_path: str) -> FastAPI:
    """Create a configured FastAPI application for the given vault."""
    app = FastAPI(title="paper-distiller web", version="1.0.0")

    # ------------------------------------------------------------------ #
    # Routes — imported here to defer fastapi dependency check             #
    # ------------------------------------------------------------------ #
    from .routes.config import router as config_router        # noqa: PLC0415
    from .routes.vault import router as vault_router          # noqa: PLC0415
    from .routes.chat import router as chat_router            # noqa: PLC0415
    from .routes.health import router as health_router        # noqa: PLC0415

    # Store vault_path in app state so routes can access it
    app.state.vault_path = vault_path

    app.include_router(config_router)
    app.include_router(vault_router)
    app.include_router(chat_router)
    app.include_router(health_router)

    # ------------------------------------------------------------------ #
    # Static files + HTML root                                             #
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        html_path = STATIC_DIR / "paper-distiller.html"
        if not html_path.exists():
            return HTMLResponse(
                "<html><body><h1>Frontend not installed</h1>"
                "<p>Run T1.4 to copy static files.</p></body></html>",
                status_code=200,
            )
        html = html_path.read_text(encoding="utf-8")
        html = _inject_vault_meta(html, vault_path)
        return HTMLResponse(html)

    # Mount static files AFTER the root route so / is handled above
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
