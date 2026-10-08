import datetime,json,pathlib,shutil,subprocess
root=pathlib.Path.cwd()
run=root/'logs/language_vision'/('experiments3_4_paired_25trials_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
run.mkdir(parents=True)
(root/'.cache/language_vision_root').write_text(str(run)+'\n')
frozen=run/'provenance/frozen/src/openpi'
shutil.copytree(root/'src/openpi',frozen,ignore=shutil.ignore_patterns('__pycache__'))
(run/'provenance/git_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
(run/'provenance/preexisting.patch').write_bytes(subprocess.check_output(['git','diff']))
processes=[]
for i in range(6):
 output=run/'servers'/f'worker{i}'
 output.mkdir(parents=True)
 command=f'source scripts/activate_env.sh train\nexport CUDA_VISIBLE_DEVICES={i}\nexport PYTHONPATH="{run}/provenance/frozen/src:$PWD:$PYTHONPATH"\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\nexec python scripts/serve_language_vision.py --port {8230+i} --output "{output}"'
 f=(output/'server.log').open('w')
 process=subprocess.Popen(['bash','-c',command],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append({'gpu':i,'port':8230+i,'pid':process.pid})
(run/'servers/processes.json').write_text(json.dumps(processes,indent=2))
(run/'protocol.json').write_text(json.dumps({'status':'pilot','scene':'tomato_sauce','layout':'original','initial_state_indices':list(range(25)), 'horizon':280,'replan_steps':5,'environment_seed':'28007+episode','rng_components':'[0,25,episode,replan]','exp3':'two demos x milk/tomato/empty language','exp4':'two demos x normal/mask_target/mask_other/mask_background/freeze_near; fixed milk language','mask':'gray127 visible object bounding box both cameras','freeze':'first planning time EEF-to-baseline-grasp-object center distance <= 0.12m; keep both camera frames thereafter','baseline_target':'first bilateral finger contact in corresponding normal milk-language rollout','lift':'object center rises >=0.03m from settled initial height','phase_sizes':[5,25]},indent=2))
print(run)
