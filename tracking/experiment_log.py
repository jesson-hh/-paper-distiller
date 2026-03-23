import os
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional


@dataclass
class Experiment:
    timestamp: str
    question: str
    method: str
    tools_used: list
    result_summary: str
    success: bool
    domain: str
    session_id: str
    tags: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)


class ExperimentLog:
    """Thread-safe JSONL-backed experiment tracker."""

    def __init__(self, log_dir: str = "research_log"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "experiments.jsonl")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._experiments: List[Experiment] = []
        self._lock = threading.Lock()
        os.makedirs(log_dir, exist_ok=True)

    def log(self, experiment: Experiment) -> None:
        with self._lock:
            self._experiments.append(experiment)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(experiment), ensure_ascii=False, default=str) + "\n")

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_result: dict,
        domain: str = "",
    ) -> None:
        """Auto-log a tool call (lightweight tracking)."""
        success = "error" not in tool_result if isinstance(tool_result, dict) else True
        summary = str(tool_result)[:200] if tool_result else ""

        exp = Experiment(
            timestamp=datetime.now().isoformat(),
            question=json.dumps(tool_input, ensure_ascii=False)[:150],
            method=tool_name,
            tools_used=[tool_name],
            result_summary=summary,
            success=success,
            domain=domain,
            session_id=self.session_id,
            raw_data={"input": tool_input, "output_preview": str(tool_result)[:500]},
        )
        self.log(exp)

    def get_session_experiments(self) -> List[Experiment]:
        with self._lock:
            return [e for e in self._experiments if e.session_id == self.session_id]

    def get_all_experiments(self) -> List[Experiment]:
        """Read all experiments from JSONL file."""
        experiments = []
        if not os.path.exists(self.log_file):
            return experiments
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        experiments.append(Experiment(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return experiments

    def query(
        self,
        domain: Optional[str] = None,
        method: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> List[Experiment]:
        results = self.get_session_experiments()
        if domain is not None:
            results = [e for e in results if e.domain == domain]
        if method is not None:
            results = [e for e in results if e.method == method]
        if success is not None:
            results = [e for e in results if e.success == success]
        return results

    def summary_stats(self) -> dict:
        exps = self.get_session_experiments()
        if not exps:
            return {"total": 0}

        methods = {}
        domains = {}
        for e in exps:
            methods[e.method] = methods.get(e.method, 0) + 1
            if e.domain:
                domains[e.domain] = domains.get(e.domain, 0) + 1

        success_count = sum(1 for e in exps if e.success)
        return {
            "total": len(exps),
            "success_rate": success_count / len(exps) if exps else 0,
            "by_method": methods,
            "by_domain": domains,
            "session_id": self.session_id,
        }

    def to_dataframe_rows(self) -> list:
        """Return session experiments as rows for Gradio Dataframe."""
        rows = []
        for e in self.get_session_experiments():
            rows.append([
                e.timestamp.split("T")[-1][:8] if "T" in e.timestamp else e.timestamp[-8:],
                e.method,
                e.domain or "-",
                e.result_summary[:80],
                "pass" if e.success else "fail",
            ])
        return rows


# Module-level singleton
_experiment_log: Optional[ExperimentLog] = None


def get_experiment_log() -> ExperimentLog:
    global _experiment_log
    if _experiment_log is None:
        _experiment_log = ExperimentLog()
    return _experiment_log
