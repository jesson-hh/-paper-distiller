import os
import json
from dataclasses import asdict
from datetime import datetime

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from tracking.experiment_log import ExperimentLog


class NotebookGenerator:
    """Generate Jupyter notebooks from experiment logs."""

    def __init__(self, experiment_log: ExperimentLog):
        self.log = experiment_log

    def generate(self, session_id: str = None) -> str:
        """Generate a Jupyter notebook and return the file path."""
        if session_id:
            experiments = [
                e for e in self.log.get_all_experiments()
                if e.session_id == session_id
            ]
        else:
            experiments = self.log.get_session_experiments()

        exp_dicts = [asdict(e) for e in experiments]
        nb = new_notebook()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }

        # ── Title cell ──
        stats = self.log.summary_stats()
        nb.cells.append(new_markdown_cell(
            f"# Math Research Session Report\n\n"
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"**Session ID**: {self.log.session_id}\n\n"
            f"**Total Experiments**: {stats.get('total', 0)}\n\n"
            f"**Success Rate**: {stats.get('success_rate', 0):.0%}\n"
        ))

        # ── Setup cell ──
        nb.cells.append(new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib\n"
            "matplotlib.rcParams['font.size'] = 12\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n\n"
            f"experiments = {json.dumps(exp_dicts, indent=2, ensure_ascii=False, default=str)}\n\n"
            "df = pd.DataFrame(experiments)\n"
            "print(f'Loaded {len(df)} experiments')\n"
            "df.head()"
        ))

        # ── Experiments by method ──
        nb.cells.append(new_markdown_cell("## Experiments by Method"))
        nb.cells.append(new_code_cell(
            "if not df.empty:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "    \n"
            "    # Count by method\n"
            "    df['method'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')\n"
            "    axes[0].set_title('Experiments by Method')\n"
            "    axes[0].set_ylabel('Count')\n"
            "    axes[0].tick_params(axis='x', rotation=45)\n"
            "    \n"
            "    # Success rate by method\n"
            "    success_rates = df.groupby('method')['success'].mean()\n"
            "    colors = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in success_rates]\n"
            "    success_rates.plot(kind='bar', ax=axes[1], color=colors)\n"
            "    axes[1].set_title('Success Rate by Method')\n"
            "    axes[1].set_ylabel('Rate')\n"
            "    axes[1].set_ylim(0, 1.1)\n"
            "    axes[1].tick_params(axis='x', rotation=45)\n"
            "    \n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ))

        # ── Domain coverage ──
        nb.cells.append(new_markdown_cell("## Domain Coverage"))
        nb.cells.append(new_code_cell(
            "if not df.empty and 'domain' in df.columns:\n"
            "    domain_counts = df[df['domain'] != '']['domain'].value_counts()\n"
            "    if not domain_counts.empty:\n"
            "        domain_counts.plot(kind='barh', color='coral', figsize=(10, 4))\n"
            "        plt.title('Experiments per Domain')\n"
            "        plt.xlabel('Count')\n"
            "        plt.tight_layout()\n"
            "        plt.show()"
        ))

        # ── Timeline ──
        nb.cells.append(new_markdown_cell("## Research Timeline"))
        nb.cells.append(new_code_cell(
            "if not df.empty:\n"
            "    df['time'] = pd.to_datetime(df['timestamp'])\n"
            "    df['minute'] = (df['time'] - df['time'].min()).dt.total_seconds() / 60\n"
            "    \n"
            "    colors = ['#2ecc71' if s else '#e74c3c' for s in df['success']]\n"
            "    plt.figure(figsize=(12, 3))\n"
            "    plt.scatter(df['minute'], [1]*len(df), c=colors, s=100, zorder=5)\n"
            "    for i, row in df.iterrows():\n"
            "        plt.annotate(row['method'][:15], (row['minute'], 1.05),\n"
            "                     rotation=45, fontsize=8, ha='left')\n"
            "    plt.xlabel('Minutes since start')\n"
            "    plt.title('Research Timeline (green=success, red=fail)')\n"
            "    plt.yticks([])\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ))

        # ── Detailed results ──
        nb.cells.append(new_markdown_cell("## Detailed Results"))
        nb.cells.append(new_code_cell(
            "if not df.empty:\n"
            "    for i, row in df.iterrows():\n"
            "        status = 'PASS' if row['success'] else 'FAIL'\n"
            "        print(f\"[{status}] {row['method']}: {row['question'][:80]}\")\n"
            "        print(f\"  -> {row['result_summary'][:120]}\")\n"
            "        print()"
        ))

        # ── References (arxiv papers) ──
        arxiv_exps = [e for e in experiments if "arxiv_search" in (e.tools_used or [])]
        if arxiv_exps:
            nb.cells.append(new_markdown_cell("## References (ArXiv Papers Searched)"))
            refs = []
            for exp in arxiv_exps:
                refs.append(f"- **{exp.question[:100]}** ({exp.domain or 'general'})")
            nb.cells.append(new_markdown_cell("\n".join(refs)))

        # ── Save ──
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filepath = os.path.join(self.log.log_dir, f"session_{date_str}.ipynb")
        with open(filepath, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return filepath
