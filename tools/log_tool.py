from datetime import datetime
from tracking import get_experiment_log, Experiment


def log_experiment(
    question: str,
    method: str,
    result_summary: str,
    success: bool,
    tools_used: list = None,
    domain: str = "",
    tags: list = None,
) -> dict:
    log = get_experiment_log()
    exp = Experiment(
        timestamp=datetime.now().isoformat(),
        question=question,
        method=method,
        tools_used=tools_used or [],
        result_summary=result_summary,
        success=success,
        domain=domain,
        session_id=log.session_id,
        tags=tags or [],
    )
    log.log(exp)

    stats = log.summary_stats()
    return {
        "logged": True,
        "experiment_id": stats["total"],
        "session_total": stats["total"],
        "session_success_rate": f"{stats['success_rate']:.0%}",
    }
