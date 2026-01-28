import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


# ============================================================
# Base sink
# ============================================================

class LogSink:

    def emit(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError


# ============================================================
# Stdout sink
# ============================================================

class StdoutSink(LogSink):

    def emit(self, event: Dict[str, Any]) -> None:
        try:
            print(json.dumps(event), flush=True)
        except Exception:
            print(
                json.dumps({
                    "event_type": "logging_failed",
                    "message": "Failed to emit log to stdout"
                }),
                file=sys.stderr,
                flush=True
            )


# ============================================================
# JSONL file sink
# ============================================================

class JsonlFileSink(LogSink):

    def __init__(
        self,
        file_path: Path,
        create_dirs: bool = True,
    ) -> None:

        self.file_path = file_path

        if create_dirs:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> None:
        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event))
                f.write("\n")
        except Exception:
            print(
                json.dumps({
                    "event_type": "logging_failed",
                    "message": f"Failed to write log to file: {self.file_path}"
                }),
                file=sys.stderr,
                flush=True
            )


# ============================================================
# Composite sink
# ============================================================

class MultiSink(LogSink):

    def __init__(self, sinks: Optional[list[LogSink]] = None) -> None:
        self.sinks = sinks or []

    def emit(self, event: Dict[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                # Never allow logging failures to affect inference
                continue