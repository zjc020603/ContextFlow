"""Serve a local experiment comparison page, with HTTP video seeking support."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from aiohttp import web

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "logs/goal_long/full_correct_context_50trials_20260924_121500"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ensure", action="store_true", help="Start in background if this viewer is not running")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "demonstrations/index.html").is_file():
        raise FileNotFoundError(root / "demonstrations/index.html")
    origin = f"http://127.0.0.1:{args.port}"
    url = origin + "/demonstrations/index.html"
    identity = {"app": "contextflow-goal-long-viewer", "root": str(root)}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def running():
        try:
            with opener.open(origin + "/_viewer_health", timeout=2) as response:
                actual = json.load(response)
        except urllib.error.URLError:
            return False
        if actual != identity:
            raise RuntimeError("This port is occupied by a different viewer")
        return True

    if args.ensure:
        if not running():
            with (root / "viewer_http.log").open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--port", str(args.port)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            for _ in range(50):
                if process.poll() is not None:
                    raise RuntimeError("Viewer exited; see viewer_http.log")
                if running():
                    (root / "viewer_service.json").write_text(
                        json.dumps({**identity, "pid": process.pid, "url": url}, indent=2) + "\n"
                    )
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Viewer readiness timed out; see viewer_http.log")
        print(url)
        return

    async def home(_request):
        raise web.HTTPFound("/demonstrations/index.html")

    async def health(_request):
        return web.json_response(identity)

    app = web.Application()
    app.router.add_get("/", home)
    app.router.add_get("/_viewer_health", health)
    # aiohttp FileResponse handles Range/HEAD, MIME types and conditional requests.
    app.router.add_static("/", root, show_index=False, follow_symlinks=False)
    print(url, flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
