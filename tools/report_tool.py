from reporting.report_generator import ReportGenerator
from tracking import get_experiment_log


def generate_report(
    title: str = None,
    format: str = "markdown",
    include_code: bool = True,
) -> dict:
    log = get_experiment_log()
    generator = ReportGenerator(log)
    filepath = generator.generate(title=title, fmt=format, include_code=include_code)

    return {
        "filepath": filepath,
        "format": format,
        "message": f"Research report saved to {filepath}",
    }
