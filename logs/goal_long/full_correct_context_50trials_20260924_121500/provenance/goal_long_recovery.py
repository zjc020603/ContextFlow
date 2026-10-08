import json, os, signal, time
from pathlib import Path
root=Path(Path('.cache/goal_long_root').read_text().strip())
controller=1108565
replacement=1296751
record=json.loads((root/'provenance/startup_recovery.json').read_text())
record.update(replacement_pid=replacement, physical_gpu=0, startup_attempt2_pid=1252480, diagnosis='Repeated startup stall materializing JAX GPU arrays for parameter hash on physical GPU5; exact underlying cause unknown. No task started on either failed attempt.')
path=root/'provenance/startup_recovery.json'
path.write_text(json.dumps(record,indent=2)+'\n')
for _ in range(600):
 if (root/'servers/gpu5/READY.json').exists():
  assert (root/'servers/gpu5/GATE_PASSED.json').exists()
  os.kill(controller,signal.SIGCONT)
  record.update(controller_dispatch_temporarily_paused=False,resumed_unix=time.time())
  path.write_text(json.dumps(record,indent=2)+'\n')
  print('Replacement passed gate; resumed controller',flush=True)
  break
 time.sleep(2)
else: raise RuntimeError('Replacement startup failed; controller remains paused')
while True:
 proc=Path(f'/proc/{controller}/stat')
 if not proc.exists() or proc.read_text().split()[2]=='Z': break
 time.sleep(5)
finished=list((root/'servers').glob('gpu*/FINISHED.json'))
assert len(finished)==8, 'Controller exited before all workers finalized'
assert len(list((root/'tasks').glob('*/*/results.json')))==20
stopped=[]
for p in sorted(root.glob('worker_gpu*.json')):
 pid=json.loads(p.read_text())['server_pid']
 command=Path(f'/proc/{pid}/cmdline')
 if command.exists() and command.read_bytes():
  cmd=command.read_bytes()
  assert b'scripts.serve_goal_long' in cmd and str(root).encode() in cmd
  os.kill(pid,signal.SIGTERM)
  stopped.append(pid)
record.update(cleanup_completed=True,finished_workers=8,finished_tasks=20,cleanup_stopped_pids=stopped,cleanup_note='Original controller retained pre-restart worker5 PID; replacement supervisor cleaned remaining owned services after all tasks and final parameter hashes completed.',ended_unix=time.time())
path.write_text(json.dumps(record,indent=2)+'\n')
print('All tasks finalized; cleanup complete',flush=True)
