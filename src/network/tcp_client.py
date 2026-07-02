import socket
import json

class TCPClient:
    """
    5-digit length prefix TCP client
    protocol: 000xx + JSON UTF-8
    """

    def __init__(self, host="127.0.0.1", port=9999):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.sock.connect((self.host, self.port))

    def _recv_n(self, n):
        data = b""
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket closed")
            data += packet
        return data

    def recv(self):
        length_bytes = self._recv_n(5)
        length = int(length_bytes.decode())
        body = self._recv_n(length)
        return json.loads(body.decode("utf-8"))

    def send(self, msg: dict):
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        length = str(len(body)).zfill(5).encode()
        self.sock.sendall(length + body)
