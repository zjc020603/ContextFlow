import socket
from unittest import mock

from scripts import serve_policy


def test_server_starts_without_resolvable_hostname(monkeypatch):
    policy = mock.Mock(metadata={"test": True})
    server = mock.Mock()
    factory = mock.Mock(return_value=server)
    resolve = mock.Mock(side_effect=socket.gaierror(-2, "Name or service not known"))
    monkeypatch.setattr(serve_policy, "create_policy", lambda args: policy)
    monkeypatch.setattr(serve_policy.socket, "gethostname", lambda: "unregistered-cluster-host")
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", resolve)
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", factory)

    serve_policy.main(serve_policy.Args(port=8765))

    resolve.assert_not_called()
    factory.assert_called_once_with(policy=policy, host="0.0.0.0", port=8765, metadata={"test": True})
    server.serve_forever.assert_called_once_with()
