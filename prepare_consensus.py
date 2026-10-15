#!/usr/bin/env python3
"""Extract only ID, consensus labels and numeric T-derivation fields; omit all report text."""
import argparse
from pathlib import Path
import pandas as pd

def extract(train_val,test):
    rows=[]
    fields={'id':'id','partition':'partition','case':'case_id','T':'t','N':'n','M':'m','サイズ不明':'size_unknown','小計_T_原発巣サイズ':'size_t','小計_T_進展範囲浸潤':'extension_t','小計_T_肺内転移':'intrapulmonary_t'}
    for file,splits in [(train_val,{'train','val'}),(test,{'test'})]:
        d=pd.read_csv(file,header=1)
        assert set(fields)<=set(d.columns)
        d=d[list(fields)].rename(columns=fields)
        assert set(d.partition)==splits
        rows.append(d)
    d=pd.concat(rows,ignore_index=True).sort_values(['partition','id']).reset_index(drop=True)
    assert len(d)==243 and not d.id.duplicated().any()
    assert set(d.size_unknown)=={0,1}
    assert not d.isna().any().any()
    assert ((d.size_t==0)==(d.size_unknown==1)).all()
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-val',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=extract(a.train_val,a.test);a.output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.output,index=False)
    print('Extracted 243 label-only consensus rows; report text and excerpts omitted.')
if __name__=='__main__':main()
