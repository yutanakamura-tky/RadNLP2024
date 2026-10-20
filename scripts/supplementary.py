from pathlib import Path
import ast,abc,math,warnings,zipfile,json
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,fbeta_score,f1_score,recall_score
from common import DATA, OUTPUT, PACKAGE, MainTaskMetric, SubTaskMetric
ROOT=DATA
OUT=OUTPUT/'supplementary';OUT.mkdir(parents=True,exist_ok=True);O=OUT;P=ROOT/'sources/RadNLP2024'
M=MainTaskMetric;z=zipfile.ZipFile(P/'dataset_for_hugging_face_hub.zip')
manifest=pd.read_csv(PACKAGE/'config/leaderboard_runs.csv')
mapping=pd.read_csv(P/'all_corpus.csv').set_index('id').case_id
prior_ci=pd.read_csv(OUTPUT/'leaderboard/Tables2_3_all_metrics_CI.csv')
s5=[];s6=[];widths=[];s8=[];s10=[];case_rows=[];changes=[];summaries=[]
for lang,name,ver,seed,nteams,ncases in [('en','english',3,42,10,9),('ja','japanese',2,43,11,24)]:
    g=pd.read_csv(z.open(f'dataset_for_hugging_face_hub/main/{name}_test.csv'))[M.COL_NAMES]
    mats={c:pd.read_csv(P/f'{c}_{lang}({ver}).csv').set_index('ID') for c in 'tnm'}
    cases=mapping.loc[g.id].to_numpy();unique=sorted(set(cases));assert len(unique)==ncases
    cdraw=np.random.default_rng(seed).integers(0,ncases,(10000,ncases))
    rdraw=np.random.default_rng(seed).integers(0,len(g),(10000,len(g)))
    items=manifest[manifest.language==lang];assert len(items)==nteams
    # Preserve the consolidated leaderboard order from manuscript tables.
    teams=items.sort_values('rank').team.tolist()
    allcorrect=[]
    for team in teams:
        item=items[items.team==team].iloc[0];col=item.matrix_column
        raw=pd.DataFrame({'id':g.id,**{c:mats[c].loc[g.id,col].to_numpy() for c in 'tnm'}})
        p=M.preprocess_submission(g,raw)
        for c in 'tnm':
            labels=sorted(g[c].unique())
            macro=f1_score(g[c],p[c],labels=labels,average='macro',zero_division=0)
            balanced=recall_score(g[c],p[c],labels=labels,average='macro',zero_division=0)
            s5.append(dict(language=lang,team=team,category=c.upper(),macro_f1=macro,balanced_accuracy=balanced,reference_labels=';'.join(labels)))
        correct=p[list('tnm')].to_numpy()==g[list('tnm')].to_numpy()
        allcorrect.append(np.column_stack([correct.all(axis=1),correct]))
        for metric,j in [('Joint',None),('T',0),('N',1),('M',2)]:
            ok=correct.all(axis=1) if j is None else correct[:,j]
            counts=np.array([ok[cases==case].sum() for case in unique]);rate=counts/9
            cb=rate[cdraw].mean(axis=1);rb=ok[rdraw].mean(axis=1)
            cl,ch=np.quantile(cb,[.025,.975],method='linear');rl,rh=np.quantile(rb,[.025,.975],method='linear')
            previous=prior_ci[(prior_ci.language==lang)&(prior_ci.team==team)&(prior_ci.granularity=='fine')&(prior_ci.metric==metric)].iloc[0]
            assert np.allclose([ok.mean(),cl,ch],[previous.accuracy,previous.ci_low,previous.ci_high],atol=1e-12)
            s6.append(dict(language=lang,team=team,metric=metric,accuracy=ok.mean(),ci_low=cl,ci_high=ch))
            widths.append(dict(language=lang,team=team,metric=metric,naive_ci_low=rl,naive_ci_high=rh,cluster_ci_low=cl,cluster_ci_high=ch,naive_width_pp=(rh-rl)*100,cluster_width_pp=(ch-cl)*100,width_ratio=(ch-cl)/(rh-rl) if rh>rl else None))
            s8.append(dict(language=lang,team=team,metric=metric,majority_correct_cases=int((counts>=5).sum()),total_cases=ncases,rate=float((counts>=5).mean())))
            for case,count in zip(unique,counts):case_rows.append(dict(language=lang,team=team,metric=metric,case_id=int(case),correct_reports=int(count),reports=9))
    pool=np.stack(allcorrect)
    for case in unique:
        subset=pool[:,cases==case,:];err=(~subset).sum(axis=(0,1));den=nteams*9
        s10.append(dict(language=lang,case_id=int(case),joint_error=err[0]/den,t_error=err[1]/den,n_error=err[2]/den,m_error=err[3]/den,denominator=den,joint_error_count=int(err[0]),t_error_count=int(err[1]),n_error_count=int(err[2]),m_error_count=int(err[3])))
for name,rows in [('S5',s5),('S6',s6),('S8',s8),('S10',s10),('team_CI_widths',widths),('team_case_counts',case_rows)]:pd.DataFrame(rows).to_csv(O/(name+'_LB.csv'),index=False)
wd=pd.DataFrame(widths)
for (lang,metric),group in wd.groupby(['language','metric'],sort=False):
    a=group.naive_width_pp.median();b=group.cluster_width_pp.median()
    summaries.append(dict(language=lang,metric=metric,median_naive_width_pp=a,median_cluster_width_pp=b,width_ratio=b/a))
pd.DataFrame(summaries).to_csv(O/'S7_LB.csv',index=False)

print(pd.DataFrame(summaries).to_string(index=False))
