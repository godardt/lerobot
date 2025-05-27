from datetime import datetime, timezone


def capture_timestamp_utc():
    return datetime.now(timezone.utc)
