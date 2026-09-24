"""Per-run file logging for the Vision4Coil processing pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def redact_source(source) -> str:
    """Return an input identifier without credentials embedded in its URL."""
    value = str(source)
    try:
        parts = urlsplit(value)
    except ValueError:
        return value

    if not parts.scheme or not parts.netloc or parts.hostname is None:
        return value

    try:
        port = parts.port
    except ValueError:
        port = None

    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    if parts.username is not None:
        host = f"***:***@{host}"

    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


class PipelineRunLog:
    """Own a uniquely named log file for one video or stream run."""

    def __init__(self, source, mode: str, log_dir="logs"):
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.perf_counter()
        self.source = redact_source(source)
        self.mode = str(mode)
        self._finished = False

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{self.started_at:%Y%m%dT%H%M%S_%fZ}_{uuid.uuid4().hex[:8]}"
        self.path = log_dir / f"pipeline_{run_id}.log"

        self.logger = logging.getLogger(f"vision4coil.pipeline.{run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = logging.FileHandler(self.path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03dZ | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self._handler = handler

        self.info(
            "pipeline_started mode=%s source=%s log_file=%s",
            self.mode,
            self.source,
            self.path,
        )

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self.logger.exception(message, *args, **kwargs)

    def finish(self, status: str, **fields):
        if self._finished:
            return

        duration_s = time.perf_counter() - self.started_monotonic
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" {details}" if details else ""
        self.info(
            "pipeline_finished status=%s duration_s=%.3f%s",
            status,
            duration_s,
            suffix,
        )
        self._finished = True
        self._handler.flush()
        self._handler.close()
        self.logger.removeHandler(self._handler)

