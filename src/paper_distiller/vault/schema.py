"""Default vault category schema.

v0.1: hardcoded. v0.2+ will read a paper-distiller.toml in the vault root for
per-vault overrides.
"""

CATEGORIES = ("articles", "techniques", "directions", "open-problems", "authors", "surveys")

DEFAULT_SCHEMA = {
    "articles": "Paper notes — one entry per paper.",
    "techniques": "Methods, proof tricks, frameworks.",
    "directions": "Research programmes / threads.",
    "open-problems": "Open problems, conjectures.",
    "authors": "Author-level distillation hubs.",
    "surveys": "Cluster / theme mini-surveys across multiple papers.",
}
