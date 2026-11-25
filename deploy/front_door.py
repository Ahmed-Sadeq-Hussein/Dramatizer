"""front_door.py | The fixed front door (Milestone 4, step 5): one port that forwards to whichever version is live.

Normally it runs as the container dramatizer-front-door (docker/site.compose.yml):

    python /app/front_door.py --bind 0.0.0.0 --listen 5005 --listen 5105 --state /state/front_door.json --by-name

and forwards to the live container by its name (dramatizer-blue or dramatizer-green, port 5005) on the Docker
network "dramatizer". It can also run on this PC, outside Docker, forwarding to the slots' ports on 127.0.0.1:

    python deploy/front_door.py --listen 5005 --listen 5105 --state history/live/front_door.json

The live site listens on both of its old ports: 5005 (the playit tunnel and the pinned clients) and 5105 (the
Cloudflare tunnel to https://app.dramatizer.dev), so neither tunnel needs changing.

It forwards raw TCP, so TLS goes straight through to the live container: clients keep seeing the API's own
certificate (SHARING.md: they check its fingerprint), and the front door never holds a key. Which version is live
is read from the state file (written by deploy/blue_green.py) for every new connection, so a switch takes effect at
once and needs no restart. Connections already open stay with the version they started on.
"""

import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "history" / "live" / "front_door.json"
CONTAINER_PORT = 5005          # the port the API listens on inside every slot's container


def _live(state_file):
    try:
        return json.loads(Path(state_file).read_text(encoding="utf-8"))["live"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def live_port(state_file):
    """The port of the live version on this PC, from the state file; None when nothing is live."""
    try:
        return int(_live(state_file)["port"])
    except (KeyError, TypeError, ValueError):
        return None


def live_target(state_file, host="127.0.0.1", by_name=False):
    """(host, port) to forward to: the live container by name on the Docker network, or its port on this PC."""
    live = _live(state_file)
    if by_name:
        return (live["container"], CONTAINER_PORT) if isinstance(live, dict) and live.get("container") else None
    port = live_port(state_file)
    return (host, port) if port is not None else None


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


def handler(state_file, host="127.0.0.1", by_name=False):
    async def handle(client_reader, client_writer):
        target = live_target(state_file, host, by_name)
        if target is None:
            client_writer.close()
            return
        try:
            server_reader, server_writer = await asyncio.open_connection(*target)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(pipe(client_reader, server_writer), pipe(server_reader, client_writer))
    return handle


async def serve(ports, state_file, host="127.0.0.1", bind="127.0.0.1", by_name=False):
    servers = [await asyncio.start_server(handler(state_file, host, by_name), bind, port) for port in ports]
    print(f"Front door on {bind}:{', '.join(map(str, ports))}, live now {live_target(state_file, host, by_name)} "
          f"({state_file})", flush=True)
    await asyncio.gather(*(server.serve_forever() for server in servers))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forward one fixed port to the live version.")
    parser.add_argument("--listen", type=int, action="append", help="a port to listen on (repeat it; default 5105)")
    parser.add_argument("--state", default=str(STATE))
    parser.add_argument("--bind", default="127.0.0.1",
                        help="the address to listen on (0.0.0.0 inside its container; Docker publishes it to "
                             "127.0.0.1 only)")
    parser.add_argument("--by-name", action="store_true",
                        help="forward to the live container by name on the Docker network (when it runs in Docker)")
    args = parser.parse_args(argv)
    asyncio.run(serve(args.listen or [5105], args.state, bind=args.bind, by_name=args.by_name))


if __name__ == "__main__":
    main()
