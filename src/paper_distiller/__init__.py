"""paper-distiller — distill arXiv research papers into an Obsidian-compatible knowledge base.

The command-line entry points (``paper-distiller-chat``, ``paper-distiller-arxiv``) are the
primary interface. The names re-exported here form the **stable public Python API** for
embedding paper-distiller in your own code; everything else under ``paper_distiller.*`` is
internal and may change between minor releases without notice.

Top-level names are imported lazily (PEP 562) so that ``import paper_distiller`` stays cheap
and never pulls in heavy/optional dependencies until a symbol is actually used.

    >>> import paper_distiller
    >>> paper_distiller.__version__
    '1.12.0'
    >>> from paper_distiller import VaultStore, LLMClient  # imported on demand
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.12.0"

# Public name -> "module:attribute" for lazy resolution via __getattr__ (PEP 562).
_LAZY_EXPORTS: dict[str, str] = {
    # Configuration
    "Config": "paper_distiller.config:Config",
    "load_config": "paper_distiller.config:load_config",
    # LLM client (OpenAI-compatible function calling + streaming)
    "LLMClient": "paper_distiller.llm:LLMClient",
    # Obsidian-compatible vault store + data model
    "VaultStore": "paper_distiller.vault:VaultStore",
    "Entry": "paper_distiller.vault:Entry",
    "slugify": "paper_distiller.vault:slugify",
    # Distillation pipeline
    "rank": "paper_distiller.distill:rank",
    "distill_article": "paper_distiller.distill:distill_article",
    "ArticleResult": "paper_distiller.distill:ArticleResult",
    "compose_survey": "paper_distiller.distill:compose_survey",
    "SurveyResult": "paper_distiller.distill:SurveyResult",
    # Proof / technique knowledge base (cross-paper RAG)
    "ProofStore": "paper_distiller.proofs:ProofStore",
    "Theorem": "paper_distiller.proofs:Theorem",
    "Technique": "paper_distiller.proofs:Technique",
    "ProofSidecar": "paper_distiller.proofs:ProofSidecar",
    # arXiv source models
    "Paper": "paper_distiller.sources:Paper",
    "ArxivPaper": "paper_distiller.sources:ArxivPaper",
    # Chat permission modes
    "PermissionMode": "paper_distiller.chat.permissions:PermissionMode",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str):
    """Lazily import and cache a public API symbol on first access (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, _, attr = target.partition(":")
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value  # cache: __getattr__ runs at most once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # let type checkers / IDEs resolve the lazily-exported names
    from .chat.permissions import PermissionMode as PermissionMode
    from .config import Config as Config
    from .config import load_config as load_config
    from .distill import ArticleResult as ArticleResult
    from .distill import SurveyResult as SurveyResult
    from .distill import compose_survey as compose_survey
    from .distill import distill_article as distill_article
    from .distill import rank as rank
    from .llm import LLMClient as LLMClient
    from .proofs import ProofSidecar as ProofSidecar
    from .proofs import ProofStore as ProofStore
    from .proofs import Technique as Technique
    from .proofs import Theorem as Theorem
    from .sources import ArxivPaper as ArxivPaper
    from .sources import Paper as Paper
    from .vault import Entry as Entry
    from .vault import VaultStore as VaultStore
    from .vault import slugify as slugify
