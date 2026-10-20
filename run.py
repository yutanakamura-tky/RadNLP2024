#!/usr/bin/env python3
"""Run the final RadNLP2024 analyses with hash-verified external inputs."""
import argparse,csv,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True,help='Root containing the externally supplied input directory tree')
    p.add_argument('--output-dir',type=Path,default=Path('outputs'))
    p.add_argument('--check-inputs-only',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parent;data=args.data_dir.resolve();out=args.output_dir.resolve()
    if not __debug__:p.error('Run without -O: integrity assertions are part of the validation.')
    if data==out or data in out.parents or out in data.parents:p.error('Input and output directories must be disjoint.')
    failures=[]
    with (root/'config/input_checksums.csv').open() as h:
        for row in csv.DictReader(h):
            f=data/row['path']
            if not f.is_file():failures.append('missing: '+row['path'])
            elif hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:failures.append('hash mismatch: '+row['path'])
    if failures:
        p.exit(2,'Input verification failed:\n'+'\n'.join(failures)+'\nSee docs/INPUTS.md; do not substitute another submission or corpus version.\n')
    print('All 75 external input files passed SHA-256 checks',flush=True)
    if args.check_inputs_only:return
    out.mkdir(parents=True,exist_ok=True)
    # Refuse stale success evidence from any earlier execution.
    (out/'verification.json').unlink(missing_ok=True)
    env=dict(os.environ,RADNLP_DATA_DIR=str(data),RADNLP_OUTPUT_DIR=str(out),PYTHONHASHSEED='0',MPLBACKEND='Agg')
    jobs=['selection','scoring_checks','leaderboard','cross_language','ensemble','supplementary','additional','proxy','confusion','subtask','submission_frequency','verify']
    start=time.time();versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','matplotlib']}
    (out/'logs').mkdir(exist_ok=True)
    for job in jobs:
        print('Running '+job,flush=True)
        with (out/'logs'/f'{job}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(root/'scripts'/f'{job}.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            print((out/'logs'/f'{job}.log').read_text(),file=sys.stderr);raise SystemExit(r.returncode)
    (out/'execution.json').write_text(json.dumps(dict(python=platform.python_version(),packages=versions,elapsed_seconds=time.time()-start,stages=jobs,status='passed'),indent=2))
    print('PASS: core numerical comparisons matched; see docs/LIMITATIONS.md for analyses outside this verification. See '+str(out/'verification.json'))
if __name__=='__main__':main()
