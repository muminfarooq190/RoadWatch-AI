import sys
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import patch

import pytest

from roadwatch_ai.alerting import AlertGate, SmsNotifier
from roadwatch_ai.config import AlertsConfig


class FakeMessages:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider down")
        return type("Message", (), {"sid": "SM123"})()


class FakeClient:
    instances = []

    def __init__(self, account_sid, auth_token):
        self.credentials = (account_sid, auth_token)
        self.messages = FakeMessages()
        self.__class__.instances.append(self)


def enabled_config():
    return AlertsConfig(
        enabled=True,
        cooldown_seconds=60,
        account_sid="AC123",
        auth_token="secret",
        from_number="+10000000000",
        to_number="+919999999999",
    )


def install_fake_twilio():
    twilio = ModuleType("twilio")
    rest = ModuleType("twilio.rest")
    rest.Client = FakeClient
    twilio.rest = rest
    return patch.dict(sys.modules, {"twilio": twilio, "twilio.rest": rest})


def test_alert_gate_suppresses_only_same_key_during_cooldown():
    gate = AlertGate(60)
    assert gate.allow("JK01AB1234", 100.0)
    assert not gate.allow("JK01AB1234", 130.0)
    assert gate.allow("DL01AB1234", 131.0)
    assert gate.allow("JK01AB1234", 160.0)


def test_notifier_does_not_import_twilio_when_disabled_or_dry_run(caplog):
    caplog.set_level("INFO")
    disabled = SmsNotifier(AlertsConfig(enabled=False))
    dry_run = SmsNotifier(enabled_config(), dry_run=True)
    arguments = {
        "plate": "JK01AB1234",
        "speed_kph": 72.4,
        "limit_kph": 50.0,
        "occurred_at": datetime(2026, 7, 23, tzinfo=timezone.utc),
        "evidence_path": "evidence.jpg",
    }
    assert disabled.send(**arguments) is False
    assert dry_run.send(**arguments) is False
    assert "plate=JK01AB1234" in caplog.text


def test_notifier_reports_missing_twilio():
    with patch.dict(sys.modules, {"twilio": None, "twilio.rest": None}):
        with pytest.raises(RuntimeError, match="Twilio is missing"):
            SmsNotifier(enabled_config())


def test_notifier_sends_expected_message(caplog):
    caplog.set_level("INFO")
    FakeClient.instances.clear()
    with install_fake_twilio():
        notifier = SmsNotifier(enabled_config())
    sent = notifier.send(
        plate="JK01AB1234",
        speed_kph=72.4,
        limit_kph=50.0,
        occurred_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        evidence_path="evidence.jpg",
    )
    assert sent is True
    client = FakeClient.instances[-1]
    assert client.credentials == ("AC123", "secret")
    assert client.messages.calls == [
        {
            "body": (
                "RoadWatch overspeed: plate=JK01AB1234, speed=72.4 km/h, "
                "limit=50.0 km/h, time=2026-07-23T00:00:00+00:00, "
                "evidence=evidence.jpg"
            ),
            "from_": "+10000000000",
            "to": "+919999999999",
        }
    ]
    assert "SMS queued with SID SM123" in caplog.text


def test_notifier_keeps_running_when_provider_fails(caplog):
    with install_fake_twilio():
        notifier = SmsNotifier(enabled_config())
    notifier._client.messages = FakeMessages(fail=True)
    assert (
        notifier.send(
            plate="UNKNOWN",
            speed_kph=80,
            limit_kph=50,
            occurred_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
            evidence_path="failed.jpg",
        )
        is False
    )
    assert "SMS delivery failed" in caplog.text
