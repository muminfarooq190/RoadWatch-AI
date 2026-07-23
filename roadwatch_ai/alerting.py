from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock

from roadwatch_ai.config import AlertsConfig

LOGGER = logging.getLogger(__name__)


class AlertGate:
    """Suppress repeated alerts for the same recognized plate during a cooldown."""

    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._last_sent: dict[str, float] = {}
        self._lock = Lock()

    def allow(self, key: str, timestamp: float) -> bool:
        with self._lock:
            previous = self._last_sent.get(key)
            if previous is not None and timestamp - previous < self._cooldown_seconds:
                return False
            self._last_sent[key] = timestamp
            return True


class SmsNotifier:
    def __init__(self, config: AlertsConfig, dry_run: bool = False) -> None:
        self._config = config
        self._dry_run = dry_run
        self._client = None
        if config.enabled and not dry_run:
            try:
                from twilio.rest import Client
            except ImportError as exc:
                raise RuntimeError(
                    "Twilio is missing. Install the project with: pip install -e ."
                ) from exc
            self._client = Client(config.account_sid, config.auth_token)

    def send(
        self,
        *,
        plate: str,
        speed_kph: float,
        limit_kph: float,
        occurred_at: datetime,
        evidence_path: str,
    ) -> bool:
        body = (
            f"RoadWatch overspeed: plate={plate}, speed={speed_kph:.1f} km/h, "
            f"limit={limit_kph:.1f} km/h, time={occurred_at.isoformat()}, "
            f"evidence={evidence_path}"
        )
        if self._dry_run or not self._config.enabled:
            LOGGER.info("SMS dry run: %s", body)
            return False

        assert self._client is not None
        try:
            message = self._client.messages.create(
                body=body,
                from_=self._config.from_number,
                to=self._config.to_number,
            )
        except Exception:
            LOGGER.exception("SMS delivery failed; the violation remains stored locally")
            return False
        LOGGER.info("SMS queued with SID %s", message.sid)
        return True
