from pathlib import Path
import ast,abc,math,warnings,contextlib,io,json,zipfile,itertools
from typing import Literal
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon,spearmanr,rankdata
from sklearn.metrics import accuracy_score,fbeta_score

from common import DATA, OUTPUT, PACKAGE, MainTaskMetric, SubTaskMetric
ROOT=DATA
OUT=OUTPUT/'cross_language';OUT.mkdir(parents=True,exist_ok=True);P=ROOT/'sources/RadNLP2024'
Metric=MainTaskMetric
z=zipfile.ZipFile(P/'dataset_for_hugging_face_hub.zip')
gold={lang:pd.read_csv(z.open(f'dataset_for_hugging_face_hub/main/{name}_test.csv'))[Metric.COL_NAMES]
      for lang,name in [('en','english'),('ja','japanese')]}
sel=pd.read_csv(OUTPUT/'selection/representative_runs.csv')
mapping=pd.read_csv(P/'all_corpus.csv').set_index('id').case_id
shared_ids=sorted(set(gold['en'].id)&set(gold['ja'].id));assert len(shared_ids)==81
shared_cases=sorted(set(mapping.loc[shared_ids]));assert len(shared_cases)==9
teams=['TMUNLPG3','sociocom','tsukurad'];case_counts={};team_counts=[];runs=[]
for lang in ('en','ja'):
    g=gold[lang].set_index('id').loc[shared_ids].reset_index()
    assert mapping.loc[g.id].value_counts().eq(9).all()
    arrays=[]
    for team in teams:
        row=sel[(sel.language==lang)&(sel.team==team)].iloc[0]
        raw=pd.read_csv(ROOT/row.prediction_file)
        pred=Metric.preprocess_submission(g,raw)
        assert pred.id.tolist()==g.id.tolist()
        err=pred[list('tnm')].ne(g[list('tnm')]).astype(int)
        err['case_id']=mapping.loc[g.id].to_numpy()
        grouped=err.groupby('case_id')[list('tnm')].sum().loc[shared_cases]
        arrays.append(grouped)
        for case,r in grouped.iterrows():
            team_counts.append(dict(language=lang,team=team,case_id=case,**{c:int(r[c]) for c in 'tnm'},reports=9))
        runs.append(row.to_dict())
    case_counts[lang]=sum(arrays)
pd.DataFrame(team_counts).to_csv(OUT/'per_team_case_error_counts.csv',index=False)
pd.DataFrame(runs).to_csv(OUT/'table7_input_runs.csv',index=False)
case_rows=[];results=[]
for c in 'tnm':
    a=case_counts['en'][c].astype(int);b=case_counts['ja'][c].astype(int);d=a-b
    # Every case-language has the same denominator (3 teams x 9 reports).
    # Integer differences preserve mathematical ties before signed ranking.
    w=wilcoxon(d.to_numpy(),zero_method='wilcox',correction=False,alternative='two-sided',method='auto')
    nonzero=d[d!=0].to_numpy();ranks=rankdata(abs(nonzero));total=ranks.sum()
    observed=min(ranks[nonzero>0].sum(),ranks[nonzero<0].sum())
    values=[sum(rank for rank,sign in zip(ranks,signs) if sign>0)
            for signs in itertools.product((-1,1),repeat=len(nonzero))]
    exact=sum(min(v,total-v)<=observed+1e-12 for v in values)/len(values)
    assert abs(exact-w.pvalue)<1e-12
    sp=spearmanr(a,b,alternative='two-sided')
    results.append(dict(category=c.upper(),en_error=a.sum()/243,ja_error=b.sum()/243,
        difference_pp=(a.sum()-b.sum())/243*100,wilcoxon_statistic=float(w.statistic),
        wilcoxon_p=float(w.pvalue),spearman_rho=float(sp.statistic),spearman_p=float(sp.pvalue),
        paired_cases=9,nonzero_pairs=len(nonzero),denominator_per_case=27))
    for case in shared_cases:
        case_rows.append(dict(category=c.upper(),case_id=case,en_errors=int(a.loc[case]),
            ja_errors=int(b.loc[case]),en_error_rate=a.loc[case]/27,
            ja_error_rate=b.loc[case]/27,difference_numerator=int(d.loc[case]),denominator=27))
df=pd.DataFrame(results);df.to_csv(OUT/'table7_recalculated.csv',index=False)
pd.DataFrame(case_rows).to_csv(OUT/'paired_case_results.csv',index=False)
print(df.to_string(index=False))
