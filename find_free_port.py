#!/usr/bin/env python3
import socket
import sys

def find_free_port(start_port=29500, end_port=65535):
    """Find a free port in the given range."""
    for port in range(start_port, end_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    free_port = find_free_port()
    if free_port:
        print(free_port)
        sys.exit(0)
    else:
        print("No free port found", file=sys.stderr)
        sys.exit(1) 