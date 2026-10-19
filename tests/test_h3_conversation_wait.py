import dataclasses
import time
from types import SimpleNamespace

from app.services import h3 as h3_module
from app.services.h3 import H3Client


def _client(monkeypatch, payload, quiet=60.0, ok=True):
    monkeypatch.setattr(h3_module, "settings",
                        dataclasses.replace(h3_module.settings, h3_conversation_quiet_s=quiet))
    client = H3Client("http://gw")
    response = SimpleNamespace(ok=ok, json=lambda: payload)
    monkeypatch.setattr(client, "_request", lambda *a, **k: response)
    return client


def test_waits_while_a_conversation_lease_is_held(monkeypatch):
    client = _client(monkeypatch, {"lease": {"backend": "h3"}, "last_ended_at": None})
    assert client.conversation_quiet_wait_s() == 60.0


def test_waits_out_the_quiet_window_after_the_last_turn(monkeypatch):
    client = _client(monkeypatch, {"lease": None, "last_ended_at": time.time() - 20})
    assert 39 < client.conversation_quiet_wait_s() <= 40


def test_proceeds_when_quiet_disabled_or_gateway_is_old(monkeypatch):
    assert _client(monkeypatch, {"lease": None, "last_ended_at": time.time() - 90}
                   ).conversation_quiet_wait_s() == 0.0
    assert _client(monkeypatch, {"lease": None}).conversation_quiet_wait_s() == 0.0
    assert _client(monkeypatch, {}, ok=False).conversation_quiet_wait_s() == 0.0
    assert _client(monkeypatch, {"lease": {"backend": "h3"}}, quiet=0
                   ).conversation_quiet_wait_s() == 0.0
