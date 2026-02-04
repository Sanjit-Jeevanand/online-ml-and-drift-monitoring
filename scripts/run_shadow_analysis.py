from datetime import datetime, timedelta, timezone
from src.monitoring.shadow_analysis import run_shadow_analysis

if __name__ == "__main__":
    # Analyse last 10 minutes (same window as monitoring)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=10)

    artifact = run_shadow_analysis(
        start_time=start,
        end_time=end,
    )

    print("Shadow analysis complete")
    print(artifact)