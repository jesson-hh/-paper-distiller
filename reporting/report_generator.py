import os
import json
from datetime import datetime

from dotenv import load_dotenv

from tracking.experiment_log import ExperimentLog

load_dotenv()


class ReportGenerator:
    """Generate structured research reports from experiment logs."""

    def __init__(self, experiment_log: ExperimentLog):
        self.log = experiment_log

    def generate(
        self,
        title: str = None,
        fmt: str = "markdown",
        include_code: bool = True,
    ) -> str:
        """Generate a research report and return the file path."""
        experiments = self.log.get_session_experiments()
        if not experiments:
            return self._save_empty_report(fmt)

        # Categorize experiments
        literature = [e for e in experiments if any(t in (e.tools_used or []) for t in ["arxiv_search"])]
        computations = [e for e in experiments if any(t in (e.tools_used or []) for t in ["symbolic_compute", "run_code"])]
        proofs = [e for e in experiments if any(t in (e.tools_used or []) for t in ["proof_assist"])]
        logged = [e for e in experiments if e.method not in ("arxiv_search", "symbolic_compute", "run_code", "proof_assist")]

        # Build data summary for Claude
        data_summary = {
            "total_experiments": len(experiments),
            "success_rate": sum(1 for e in experiments if e.success) / len(experiments),
            "literature_searches": [
                {"question": e.question, "result": e.result_summary[:200], "domain": e.domain}
                for e in literature
            ],
            "computations": [
                {"question": e.question, "result": e.result_summary[:200], "method": e.method}
                for e in computations
            ],
            "proofs": [
                {"question": e.question, "result": e.result_summary[:200], "success": e.success}
                for e in proofs
            ],
            "key_findings": [
                {"question": e.question, "result": e.result_summary, "tags": e.tags}
                for e in logged
            ],
        }

        # Determine format instruction
        format_instruction = (
            "Write in LaTeX format with proper \\section{}, \\begin{theorem}, etc."
            if fmt == "latex"
            else "Write in Markdown format with ## headers and $...$ for inline math."
        )

        synthesis_prompt = (
            f"Based on the following research session data, write a structured "
            f"mathematical research report.\n\n"
            f"## Session Data\n"
            f"```json\n{json.dumps(data_summary, indent=2, ensure_ascii=False, default=str)}\n```\n\n"
            f"## Report Requirements\n"
            f"- {format_instruction}\n"
            f"- Structure: Title, Abstract, Introduction, Literature Review, "
            f"Methods & Computations, Results, Discussion (open questions, next steps), References\n"
            f"- Use precise mathematical notation\n"
            f"- Cite papers by title and arxiv ID where available\n"
            f"- Be scholarly but concise\n"
            f"{'- Include key code snippets for reproducibility' if include_code else ''}\n"
        )

        if title:
            synthesis_prompt += f"\n**Report Title**: {title}\n"

        # Call LLM for synthesis
        from llm import get_client
        client = get_client()
        result = client.chat(
            system="You are a scientific report writer specializing in mathematics.",
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=8192,
        )
        report_content = ""
        for block in result["content_blocks"]:
            if block["type"] == "text":
                report_content += block["text"]

        # Save
        return self._save_report(report_content, fmt)

    def _save_report(self, content: str, fmt: str) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        ext = "tex" if fmt == "latex" else "md"
        filepath = os.path.join(self.log.log_dir, f"report_{date_str}.{ext}")
        os.makedirs(self.log.log_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def _save_empty_report(self, fmt: str) -> str:
        content = (
            "# Research Report\n\n"
            "No experiments were recorded in this session. "
            "Use the agent to search papers, compute, prove theorems, "
            "or run code experiments first."
        )
        return self._save_report(content, fmt)
