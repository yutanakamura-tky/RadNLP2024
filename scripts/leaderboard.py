from pathlib import Path
import ast,abc,math,warnings,zipfile,hashlib
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,fbeta_score
from common import DATA, OUTPUT, PACKAGE, MainTaskMetric, SubTaskMetric
ROOT=DATA
OUT=OUTPUT/'leaderboard';OUT.mkdir(parents=True,exist_ok=True);O=OUT;P=ROOT/'sources/RadNLP2024'
M=MainTaskMetric;z=zipfile.ZipFile(P/'dataset_for_hugging_face_hub.zip')
manifest=pd.read_csv(PACKAGE/'config/leaderboard_runs.csv')
mapping=pd.read_csv(P/'all_corpus.csv').set_index('id').case_id
output=[];case_results=[];inputs=[]
for lang,name,ver,ti,seed,ncases,nteams in [('en','english',3,1,42,9,10),('ja','japanese',2,2,43,24,11)]:
    g=pd.read_csv(z.open(f'dataset_for_hugging_face_hub/main/{name}_test.csv'))[M.COL_NAMES]
    mats={c:pd.read_csv(P/f'{c}_{lang}({ver}).csv').set_index('ID') for c in 'tnm'}
    cases=mapping.loc[g.id].to_numpy();unique=sorted(set(cases));assert len(unique)==ncases
    assert pd.Series(cases).value_counts().eq(9).all()
    draws=np.random.default_rng(seed).integers(0,ncases,size=(10000,ncases))
    rows=manifest[manifest.language==lang].sort_values("rank");assert len(rows)==nteams
    for row in rows.itertuples():
        rank=int(row.rank);team=row.team
        item=manifest[(manifest.language==lang)&(manifest.team==team)].iloc[0];col=item.matrix_column
        raw=pd.DataFrame({'id':g.id,**{c:mats[c].loc[g.id,col].to_numpy() for c in 'tnm'}})
        p=M.preprocess_submission(g,raw);assert p.id.tolist()==g.id.tolist()
        for granularity in ['fine','coarse']:
            pred=p[list('tnm')].to_numpy();ref=g[list('tnm')].to_numpy()
            if granularity=='coarse':
                pred=np.column_stack([p[c].map(lambda v:M._to_coarse_label(c,v)) for c in 'tnm'])
                ref=np.column_stack([g[c].map(lambda v:M._to_coarse_label(c,v)) for c in 'tnm'])
            correct=pred==ref
            for metric,j in [('Joint',None),('T',0),('N',1),('M',2)]:
                ok=correct.all(axis=1) if j is None else correct[:,j]
                percase=np.array([ok[cases==case].mean() for case in unique])
                point=float(ok.mean());assert abs(point-percase.mean())<1e-12
                boot=percase[draws].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],method='linear')
                output.append(dict(language=lang,rank=rank,team=team,granularity=granularity,metric=metric,
                    accuracy=point,ci_low=float(lo),ci_high=float(hi),correct_reports=int(ok.sum()),
                    total_reports=len(g),imaging_cases=ncases,bootstrap_resamples=10000,seed=seed))
                for case,acc in zip(unique,percase):case_results.append(dict(language=lang,team=team,
                    granularity=granularity,metric=metric,case_id=int(case),correct_reports=int(round(acc*9)),total_reports=9))
        inputs.append(item.to_dict())
df=pd.DataFrame(output);assert len(df)==168
df.to_csv(O/'Tables2_3_all_metrics_CI.csv',index=False)
pd.DataFrame(case_results).to_csv(O/'per_case_correct_counts.csv',index=False)
pd.DataFrame(inputs).to_csv(O/'LB_input_manifest.csv',index=False)
print(df[(df.granularity=='fine')&(df.metric=='Joint')].to_string(index=False))
