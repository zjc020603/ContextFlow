"""Run one paired phase against already gated local servers; fail closed on any job error."""

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path


def main(args):
    root = Path(args.run).resolve()
    workers = json.loads((root / "servers/processes.json").read_text())
    if args.workers is not None:
        workers = [w for w in workers if w["gpu"] in args.workers]
    assert workers
    deadline = time.monotonic() + 1200
    while not all((root / "servers" / f"worker{w['gpu']}" / "READY.json").exists() for w in workers):
        if time.monotonic() > deadline:
            raise TimeoutError("Servers did not pass gates")
        time.sleep(2)
    jobs = []
    for demo in (args.demo,) if args.demo else ("milk", "tomato_sauce"):
        if args.phase == "language":
            jobs.extend((demo, language, "normal") for language in ("milk", "tomato_sauce", "empty"))
        else:
            jobs.extend((demo, "milk", v) for v in ("mask_target", "mask_other", "mask_background", "freeze_near"))

    def worker(index):
        for demo, language, vision in jobs[index :: len(workers)]:
            label = f"{args.phase}_{demo}_{language}_{vision}_{args.start}_{args.stop}"
            log = root / "jobs" / (label + ".log")
            log.parent.mkdir(exist_ok=True)
            command = [
                "examples/libero/.venv/bin/python",
                "-m",
                "examples.libero.language_vision_experiment",
                "--port",
                str(workers[index]["port"]),
                "--output",
                str(root),
                "--demo",
                demo,
                "--language",
                language,
                "--vision",
                vision,
                "--start",
                str(args.start),
                "--stop",
                str(args.stop),
            ]
            with log.open("w") as f:
                result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{label} failed: {log}")
            print("FINISHED", label, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [pool.submit(worker, i) for i in range(len(workers))]
        for future in futures:
            future.result()
    (root / f"{args.phase}_{args.start}_{args.stop}{'_' + args.demo if args.demo else ''}_DONE.json").write_text(
        json.dumps({"jobs": len(jobs), "episodes": len(jobs) * (args.stop - args.start)})
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--phase", choices=("language", "vision"), required=True)
    p.add_argument("--workers", type=int, nargs="+")
    p.add_argument("--demo", choices=("milk", "tomato_sauce"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=5)
    main(p.parse_args())
