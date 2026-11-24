"""front_door.py | The fixed front door (Milestone 4, step 5): one port that forwards to whichever version is live.

    python deploy/front_door.py --listen 5105 --state history/front_door.json

It forwards raw TCP, so TLS goes straight through to the live container: clients keep seeing the API's own
certificate (SHARING.md: they check its fingerprint), and the front door never holds a key. Which port is live is
read from the state file (written by deploy/blue_green.py) for every new connection, so a switch takes effect at
once and needs no restart. Connections already open stay with the version they started on.
"""

import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "history" / "front_door.json"


def live_port(state_file):
    """The port of the live version, from the state file; None when nothing is live."""
    try:
        return int(json.loads(Path(state_file).read_text(encoding="utf-8"))["live"]["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


def handler(state_file, host):
    async def handle(client_reader, client_writer):
        port = live_port(state_file)
        if port is None:
            client_writer.close()
            return
        try:
            server_reader, server_writer = await asyncio.open_connection(host, port)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(pipe(client_reader, server_writer), pipe(server_reader, client_writer))
    return handle


async def serve(listen, state_file, host="127.0.0.1"):
    server = await asyncio.start_server(handler(state_file, host), "127.0.0.1", listen)
    print(f"Front door on 127.0.0.1:{listen}, live port now {live_port(state_file)} ({state_file})", flush=True)
    async with server:
        await server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forward one fixed port to the live version.")
    parser.add_argument("--listen", type=int, default=5105)
    parser.add_argument("--state", default=str(STATE))
    args = parser.parse_args(argv)
    asyncio.run(serve(args.listen, args.state))


if __name__ == "__main__":
    main()
