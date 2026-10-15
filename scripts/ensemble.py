from pathlib import Path
import ast,abc,math,warnings,zipfile,hashlib
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,fbeta_score
from common import DATA, OUTPUT, PACKAGE, MainTaskMetric, SubTaskMetric
ROOT=DATA
OUT=OUTPUT/'ensemble';OUT.mkdir(parents=True,exist_ok=True);O=OUT;P=ROOT/'sources/RadNLP2024'
Metric=MainTaskMetric;z=zipfile.ZipFile(P/'dataset_for_hugging_face_hub.zip')
sel=pd.read_csv(OUTPUT/'selection/representative_runs.csv')
mapping=pd.read_csv(P/'all_corpus.csv').set_index('id').case_id
results=[];best=[];ties=[];report_rows=[];case_rows=[];inputs=[]
for lang,name,seed,nteams,ncases in [('en','english',42,10,9),('ja','japanese',43,11,24)]:
    g=pd.read_csv(z.open(f'dataset_for_hugging_face_hub/main/{name}_test.csv'))[Metric.COL_NAMES]
    rows=sel[sel.language==lang].copy();rows['has_val']=rows.validation_coverage>0
    # Measured validation (including zero) precedes missing validation.
    rows=rows.sort_values(['has_val','validation_joint_fine','datetime','team'],ascending=[False,False,True,True])
    arrays=[]
    for priority,row in enumerate(rows.itertuples(),1):
        f=ROOT/row.prediction_file;assert hashlib.sha256(f.read_bytes()).hexdigest()==row.sha256
        raw=pd.read_csv(f);p=Metric.preprocess_submission(g,raw)
        assert p.id.tolist()==g.id.tolist();arrays.append(p[list('tnm')].to_numpy())
        inputs.append(dict(language=lang,tie_priority=priority,team=row.team,submission_id=row.submission_id,
            rule=row.rule,validation_joint_fine=row.validation_joint_fine,datetime=row.datetime,
            prediction_file=row.prediction_file,sha256=row.sha256))
    arr=np.stack(arrays);truth=g[list('tnm')].to_numpy();correct=arr==truth
    assert arr.shape==(nteams,len(g),3)
    voted=np.empty(truth.shape,dtype=object)
    for i in range(len(g)):
        for j,c in enumerate('tnm'):
            values=arr[:,i,j];counts=pd.Series(values).value_counts(dropna=False)
            winners=set(counts[counts==counts.max()].index)
            winner_index=next(k for k,v in enumerate(values) if v in winners)
            voted[i,j]=values[winner_index]
            if len(winners)>1:
                ties.append(dict(language=lang,id=int(g.id.iloc[i]),category=c.upper(),
                    tied_labels=';'.join(sorted(winners)),votes_per_label=int(counts.max()),
                    winning_team=rows.iloc[winner_index].team,winning_label=values[winner_index],
                    winner_has_validation=bool(rows.iloc[winner_index].has_val)))
    plurality_components=voted==truth;oracle_components=correct.any(axis=0)
    plurality_joint=plurality_components.all(axis=1)
    # Joint oracle requires one team correct on all T/N/M within a report.
    oracle_joint=correct.all(axis=2).any(axis=0)
    assert (oracle_joint<=oracle_components.all(axis=1)).all()
    cases=mapping.loc[g.id].to_numpy();unique=sorted(set(cases));assert len(unique)==ncases
    assert pd.Series(cases).value_counts().eq(9).all()
    draws=np.random.default_rng(seed).integers(0,ncases,size=(10000,ncases))
    for metric,j in [('Joint',None),('T',0),('N',1),('M',2)]:
        for method in ('Plurality','Oracle'):
            ok=(plurality_joint if method=='Plurality' else oracle_joint) if j is None else (plurality_components[:,j] if method=='Plurality' else oracle_components[:,j])
            percase=np.array([ok[cases==case].mean() for case in unique])
            boot=percase[draws].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],method='linear')
            results.append(dict(language=lang,metric=metric,ensemble=method,accuracy=ok.mean(),
                ci_low=lo,ci_high=hi,majority_correct_cases=int((percase>0.5).sum()),total_cases=ncases,teams=nteams,
                correct_reports=int(ok.sum()),total_reports=len(ok),bootstrap_seed=seed))
            for case,rate in zip(unique,percase):
                case_rows.append(dict(language=lang,metric=metric,ensemble=method,case_id=case,
                    correct_reports=int(ok[cases==case].sum()),total_reports=9,majority_correct=bool(rate>0.5)))
            for report,case,v in zip(g.id,cases,ok):
                report_rows.append(dict(language=lang,metric=metric,ensemble=method,id=int(report),case_id=int(case),correct=bool(v)))
        individual=correct.all(axis=2).mean(axis=1) if j is None else correct[:,:,j].mean(axis=1)
        best.append(dict(language=lang,metric=metric,best_individual_accuracy=float(individual.max()),
             teams=';'.join(rows.iloc[k].team for k in np.flatnonzero(individual==individual.max()))))
    pd.DataFrame({'id':g.id,**{c:voted[:,j] for j,c in enumerate('tnm')}}).to_csv(OUT/f'plurality_predictions_{lang}.csv',index=False)
df=pd.DataFrame(results);df.to_csv(OUT/'tableS11_recalculated.csv',index=False)
df[df.metric!='Joint'].to_csv(OUT/'table8_recalculated.csv',index=False)
pd.DataFrame(best).to_csv(OUT/'best_representative_individual.csv',index=False)
pd.DataFrame(inputs).to_csv(OUT/'ensemble_input_runs.csv',index=False)
pd.DataFrame(ties).to_csv(OUT/'plurality_tie_checks.csv',index=False)
pd.DataFrame(report_rows).to_csv(OUT/'report_success.csv',index=False)
pd.DataFrame(case_rows).to_csv(OUT/'case_majority_success.csv',index=False)
assert len(df)==16
print(df.to_string(index=False));print('BEST',pd.DataFrame(best).to_string(index=False))
print('TIES',pd.DataFrame(ties).groupby(['language','category']).size().to_dict())
print('fallback tie winners',sum(not t['winner_has_validation'] for t in ties))
