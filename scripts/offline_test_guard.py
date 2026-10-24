"""Test-only outbound-network fence. Never imported by the production application."""
import ipaddress
import socket


def install():
    original_connect=socket.socket.connect
    original_connect_ex=socket.socket.connect_ex
    original_getaddrinfo=socket.getaddrinfo

    def permitted(address):
        if not isinstance(address,tuple):
            return  # Unix socket (local test transports)
        host=str(address[0])
        if host=='localhost':
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError('OFFLINE_TEST_NETWORK_BLOCKED: outbound network is forbidden')

    def connect(sock,address):
        permitted(address)
        return original_connect(sock,address)

    def connect_ex(sock,address):
        permitted(address)
        return original_connect_ex(sock,address)

    def getaddrinfo(host,*args,**kwargs):
        permitted((host,0))
        return original_getaddrinfo(host,*args,**kwargs)

    socket.socket.connect=connect
    socket.socket.connect_ex=connect_ex
    socket.getaddrinfo=getaddrinfo


def install_cloud_fakes():
    """Deterministic test credentials: never call ADC or probe instance identity."""
    from unittest.mock import patch
    from google.auth.credentials import AnonymousCredentials

    def fake_credentials(*args, **kwargs):
        return AnonymousCredentials(), 'offline-test-project'

    # Apply before test module collection, not after imports in an autouse fixture.
    patch('google.auth.default', fake_credentials).start()
    patch('google.auth._default.default', fake_credentials).start()
    patch('google.auth.compute_engine._metadata.ping', return_value=False).start()
