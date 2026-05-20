"""OAI-PMH incremental sync against export.arxiv.org/oai2.

arxiv's OAI server gives 1000 records per resumption-token request, with an
implicit ~15s server-side throttle between batches. Sickle handles resumption
tokens automatically; we just iterate.

Resilience: arxiv's OAI endpoint redirects to ``oaipmh.arxiv.org`` mid-stream
which occasionally fails with SSL EOF (especially over flaky/CN networks).
We catch those and resume from the last successfully-ingested record's
datestamp via a new ListRecords call. Resumption tokens are server-side and
get invalidated on connection drop — restarting from a date is the best we
can do without a true server-side replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .oai_record import record_to_paper
from .store import Store


_OAI_ENDPOINT = "https://export.arxiv.org/oai2"


@dataclass
class SyncResult:
    n_seen: int = 0
    n_inserted: int = 0
    n_deleted: int = 0
    duration_sec: float = 0.0
    n_ssl_retries: int = 0


def _is_transient_oai_error(exc: Exception) -> bool:
    """Network-level errors we should retry on."""
    s = f"{type(exc).__name__} {exc}".lower()
    return any(needle in s for needle in (
        "sslerror", "ssl: unexpected_eof", "ssl_eof",
        "connectionerror", "connection aborted", "connection reset",
        "connection broken", "max retries exceeded", "readtimeout",
        "timeout",
    ))


def _make_sickle_client():
    from sickle import Sickle
    return Sickle(_OAI_ENDPOINT)


def sync(
    store: Store,
    since: str | None = None,
    metadata_prefix: str = "arXiv",
    set_spec: str | None = None,
    progress_cb=None,
    sickle_client=None,
    max_ssl_retries: int = 8,
) -> SyncResult:
    """Pull all records updated since `since` (ISO date) into the store.

    Auto-resumes on transient SSL / network errors up to ``max_ssl_retries``
    times. Each retry re-creates the Sickle client and restarts ListRecords
    from the latest checkpoint date.

    Arguments:
        store: open Store
        since: ISO date string (e.g. "2024-01-15") or None to use last_sync from state
        metadata_prefix: 'arXiv' for arxiv-specific schema; 'oai_dc' for Dublin Core
        set_spec: optional category filter (e.g. "cs"); None = all
        sickle_client: optional injected client for tests
        max_ssl_retries: how many times to retry after a transient network error
    """
    import sys
    import time

    state = store.load_state()
    from_date = since or state.get("last_sync")

    # Strip ISO timestamp suffix if present — OAI from= expects just the date.
    def _normalize_date(d):
        if d is None:
            return None
        return d.split("T")[0] if "T" in d else d

    current_from = _normalize_date(from_date)
    use_injected_client = sickle_client is not None

    t0 = time.monotonic()
    result = SyncResult()
    BATCH = 1000
    latest_datestamp: str | None = None

    for attempt in range(max_ssl_retries + 1):
        # Build the per-attempt client + query
        client = sickle_client if use_injected_client else _make_sickle_client()
        kwargs = {"metadataPrefix": metadata_prefix}
        if current_from:
            kwargs["from"] = current_from
        if set_spec:
            kwargs["set"] = set_spec

        batch: list = []
        try:
            records = client.ListRecords(**kwargs)
            for record in records:
                result.n_seen += 1
                ds = getattr(record.header, "datestamp", None) if hasattr(record, "header") else None
                if ds:
                    latest_datestamp = ds
                if record.deleted:
                    result.n_deleted += 1
                    continue
                paper = record_to_paper(record)
                if paper is None:
                    continue
                batch.append(paper)
                if len(batch) >= BATCH:
                    store.upsert_many(batch)
                    result.n_inserted += len(batch)
                    batch.clear()
                    store.save_state({
                        "last_sync": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    if progress_cb:
                        progress_cb(result.n_seen, result.n_inserted)

            # Iteration finished cleanly
            if batch:
                store.upsert_many(batch)
                result.n_inserted += len(batch)
            break

        except Exception as e:
            # Persist whatever we already have in this batch
            if batch:
                store.upsert_many(batch)
                result.n_inserted += len(batch)

            if attempt >= max_ssl_retries or not _is_transient_oai_error(e):
                store.save_state({
                    "last_sync": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                result.duration_sec = time.monotonic() - t0
                raise

            # Resume from the latest record we saw, or one day before to
            # be safe against missed records right at the boundary.
            resume_from = latest_datestamp or current_from
            print(
                f"  [oai] transient error ({type(e).__name__}); resuming from "
                f"{resume_from} (retry {attempt + 1}/{max_ssl_retries})",
                file=sys.stderr,
            )
            current_from = resume_from
            result.n_ssl_retries += 1
            # Exponential backoff: 5s, 10s, 20s, 40s, 80s, 160s, 320s, 600s
            backoff = min(600, 5 * (2 ** attempt))
            time.sleep(backoff)
            # When using injected client (tests), don't loop — let the test
            # framework drive multiple sync() calls.
            if use_injected_client:
                store.save_state({
                    "last_sync": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                result.duration_sec = time.monotonic() - t0
                raise

    store.save_state({
        "last_sync": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    result.duration_sec = time.monotonic() - t0
    return result
