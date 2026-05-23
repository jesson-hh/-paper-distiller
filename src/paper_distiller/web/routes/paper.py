"""GET /paper/{arxiv_id}.pdf — serve cached or freshly downloaded arXiv PDF.

Contract
--------
- 400  arxiv_id fails validation
- 502  download failure (non-2xx from arXiv or network error)
- 200  application/pdf with Cache-Control: public, max-age=86400
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from paper_distiller.web.pdf_cache import get_or_download_pdf

router = APIRouter()


@router.get("/paper/{arxiv_id}.pdf")
async def serve_paper_pdf(
    arxiv_id: str,
    request: Request,
    vault_path: str = Query(default=""),
) -> FileResponse:
    """Return the arXiv PDF for *arxiv_id*, downloading and caching as needed.

    Query params
    ------------
    vault_path : str
        Absolute path to the vault root.  If not supplied the value from
        ``app.state.vault_path`` is used (set by :func:`~paper_distiller.web.server.create_app`).
    """
    vp = vault_path or getattr(request.app.state, "vault_path", "")
    if not vp:
        raise HTTPException(status_code=400, detail="vault_path is required")

    vault = Path(vp)

    try:
        pdf_path = get_or_download_pdf(arxiv_id, vault)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="could not fetch PDF") from exc

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Cache-Control": "public, max-age=86400"},
    )
