"""Run held-out tasks first, then the remaining tasks; isolated per-task RNG streams."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import logging
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]


class Worker:
    def __init__(self, root, gpu, external_pid=None):
        self.root = root
        self.gpu = gpu
        self.port = 8130 + gpu
        self.pid = external_pid
        self.proc = None
        self.directory = root / "servers" / f"gpu{gpu}"
        self.env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        self.affinity = f"{gpu*24}-{gpu*24+23}"

    def command(self, environment, args):
        code = f"source scripts/activate_env.sh {environment}\nexec " + shlex.join(args)
        return ["taskset", "-c", self.affinity, "bash", "-c", code]

    def start(self):
        if self.pid is None:
            log = (self.root / f"server_gpu{self.gpu}.log").open("w")
            self.proc = subprocess.Popen(
                self.command(
                    "train",
                    [
                        "python",
                        "-m",
                        "scripts.serve_goal_long",
                        "--output",
                        str(self.directory),
                        "--manifest",
                        str(self.root / "task_manifest.json"),
                        "--port",
                        str(self.port),
                    ],
                ),
                cwd=REPO,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            self.pid = self.proc.pid
        (self.root / f"worker_gpu{self.gpu}.json").write_text(
            json.dumps(
                {"gpu": self.gpu, "server_pid": self.pid, "port": self.port, "cpu_affinity": self.affinity}, indent=2
            )
            + "\n"
        )
        for _ in range(600):
            if (self.directory / "READY.json").exists():
                return
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"GPU {self.gpu} server exited")
            if not Path(f"/proc/{self.pid}").exists():
                raise RuntimeError("External server exited")
            time.sleep(2)
        raise TimeoutError("Model server gate timeout")

    def task(self, task, trials):
        logging.info("GPU %d starts task %d %s", self.gpu, task["dataset_task_index"], task["instruction"])
        logpath = self.root / "clients" / f"task{task['dataset_task_index']:02d}.log"
        logpath.parent.mkdir(exist_ok=True)
        with logpath.open("w") as log:
            subprocess.run(
                self.command(
                    "libero",
                    [
                        "python",
                        "-m",
                        "examples.libero.goal_long_eval",
                        "--root",
                        str(self.root),
                        "--task",
                        str(task["dataset_task_index"]),
                        "--port",
                        str(self.port),
                        "--trials",
                        str(trials),
                    ],
                ),
                cwd=REPO,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result = json.loads((self.root / "tasks" / task["task_key"] / "results.json").read_text())
        logging.info(
            "DONE task %d %s: %d/%d", task["dataset_task_index"], task["split"], result["successes"], result["episodes"]
        )
        return result

    def finish(self):
        code = f"from openpi_client.websocket_client_policy import WebsocketClientPolicy; print(WebsocketClientPolicy('127.0.0.1',{self.port}).infer({{'benchmark_control':'finalize'}}))"
        with (self.root / f"finalize_gpu{self.gpu}.log").open("w") as log:
            subprocess.run(
                self.command("libero", ["python", "-c", code]),
                cwd=REPO,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def stop(self):
        if self.pid is None:
            return
        path = Path(f"/proc/{self.pid}/cmdline")
        if path.exists():
            command = path.read_bytes()
            if b"scripts.serve_goal_long" not in command:
                raise RuntimeError("PID no longer identifies our server")
            os.kill(self.pid, signal.SIGTERM)
        if self.proc is not None:
            self.proc.wait(timeout=60)


def run(args):
    root = Path(args.root).resolve()
    tasks = json.loads((root / "task_manifest.json").read_text())["tasks"]
    unseen = sorted([t for t in tasks if t["split"] == "unseen"], key=lambda t: t["dataset_task_index"], reverse=True)
    workers = [Worker(root, i, args.gpu0_pid if i == 0 else None) for i in range(8)]
    status = {
        "protocol": "goal_long_full_correct_context_v1",
        "status": "unseen_running",
        "trials_per_task": args.trials,
        "unseen_first": True,
        "started_unix": time.time(),
        "checkpoint": "/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999",
    }

    def save():
        (root / "run.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:

            def first(worker, task):
                worker.start()
                return worker.task(task, args.trials)

            futures = [pool.submit(first, w, t) for w, t in zip(workers[:4], unseen, strict=True)]
            heldout = [f.result() for f in as_completed(futures)]
        (root / "unseen_results.json").write_text(json.dumps(heldout, indent=2) + "\n")
        status.update(status="seen_running", unseen_complete=True)
        save()
        logging.info("ALL FOUR UNSEEN TASKS COMPLETE. Starting 16 seen tasks.")
        pending = queue.Queue()
        for task in sorted([t for t in tasks if t["split"] == "seen"], key=lambda t: t["max_steps"], reverse=True):
            pending.put(task)

        def rest(worker):
            if worker.gpu >= 4:
                worker.start()
            while True:
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    return
                worker.task(task, args.trials)
                pending.task_done()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(rest, w) for w in workers]
            for f in as_completed(futures):
                f.result()
        for worker in workers:
            worker.finish()
        status.update(status="rollouts_complete", ended_unix=time.time())
        save()
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        for worker in workers:
            worker.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--gpu0-pid", type=int)
    p.add_argument("--trials", type=int, default=50)
    run(p.parse_args())
