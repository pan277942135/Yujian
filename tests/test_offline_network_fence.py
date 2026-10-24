"""Verify forbidden addresses are rejected before DNS or socket access."""
import socket
import pytest

@pytest.mark.parametrize('host',['169.254.169.254','metadata.google.internal','telemetry.example.com','8.8.8.8'])
def test_external_lookup_is_blocked_before_resolution(host):
    with pytest.raises(AssertionError,match='OFFLINE_TEST_NETWORK_BLOCKED'):
        socket.getaddrinfo(host,80)


def test_external_connect_is_blocked_before_network():
    with socket.socket() as connection:
        with pytest.raises(AssertionError,match='OFFLINE_TEST_NETWORK_BLOCKED'):
            connection.connect(('169.254.169.254',80))


def test_adc_is_deterministic_and_never_calls_metadata(monkeypatch):
    import google.auth
    from google.auth.credentials import AnonymousCredentials
    from google.auth.compute_engine import _metadata
    monkeypatch.setattr(_metadata,'ping',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('must not probe metadata')))
    credentials,project=google.auth.default()
    assert isinstance(credentials,AnonymousCredentials)
    assert project=='offline-test-project'
