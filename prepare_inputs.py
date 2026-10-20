#!/usr/bin/env python3
"""Create a label-only, byte-verified input bundle from the archived organizer files."""
import argparse,csv,hashlib,io,shutil,zipfile
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    source=a.source_dir.resolve();out=a.output_dir.resolve();root=Path(__file__).resolve().parent
    if out==source or source in out.parents or out in source.parents:p.error('Source and destination must be disjoint.')
    rows=list(csv.DictReader((root/'config/input_checksums.csv').open()))
    for row in rows:
        if row['path']=='annotations/main_consensus_proxy.csv':
            f=source/row['path'];dst=out/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            if not f.is_file():raise FileNotFoundError('First generate annotations/main_consensus_proxy.csv with prepare_consensus.py; see docs/INPUTS.md')
            shutil.copyfile(f,dst)
            if hashlib.sha256(dst.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Consensus hash mismatch')
            continue
        rel=row['path'];src=source/rel;dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if rel.endswith('/dataset_for_hugging_face_hub.zip'):
            with zipfile.ZipFile(src) as zi,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as zo:
                for f in sorted(zi.namelist()):
                    if f.startswith('dataset_for_hugging_face_hub/') and f.endswith('.csv'):
                        g=pd.read_csv(zi.open(f));buf=io.StringIO();g.drop(columns=['text'],errors='ignore').to_csv(buf,index=False)
                        info=zipfile.ZipInfo(f,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;zo.writestr(info,buf.getvalue())
        elif rel.endswith('/all_corpus.csv'):pd.read_csv(src)[['id','case_id']].to_csv(dst,index=False)
        else:shutil.copyfile(src,dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Prepared hash mismatch: '+rel)
    print(f'Prepared {len(rows)} label-only input files. No report text was copied.')
if __name__=='__main__':main()
