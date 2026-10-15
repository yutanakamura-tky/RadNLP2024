"""Count distinct submissions and verify the resolved Japanese duplicate-record discrepancy."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import PACKAGE,OUTPUT
O=OUTPUT/'submission_frequency';O.mkdir(parents=True,exist_ok=True)
r=pd.read_csv(PACKAGE/'config/submission_registry.csv');lb=pd.read_csv(OUTPUT/'leaderboard/Tables2_3_all_metrics_CI.csv');sub=pd.read_csv(OUTPUT/'subtask/Tables4_5_subtask_LB.csv')
results=[];teamcounts=[]
for task,lang in [('main','en'),('main','ja'),('sub','en'),('sub','ja')]:
    scores=(lb[(lb.language==lang)&(lb.granularity=='fine')&(lb.metric=='Joint')].set_index('team').accuracy if task=='main' else sub[sub.language==lang].set_index('team').overall_micro_f2)
    if task=='sub' and lang=='ja':
        separate=pd.read_csv(OUTPUT/'subtask/Japanese_TMUNLPG3_metadata_scores.csv')
        scores=pd.concat([scores,pd.Series({'TMUNLPG3':float(separate['Overall micro F2.0'].iloc[0])})])
    g=r[(r.task==task)&(r.language==lang)&r.team.isin(scores.index)&(r.status==3)&r.private_primary.notna()]
    variants=['selected_plus_asukai_exception','selected_only'] if task=='main' and lang=='ja' else ['selected_only']
    for rule in variants:
        chosen=g[g.selected|((g.team=='asukai89')&(rule=='selected_plus_asukai_exception'))].drop_duplicates(['team','submission_id'])
        n=chosen.groupby('team').submission_id.nunique();a=scores.reindex(n.index)
        sp=spearmanr(n,a)
        results.append(dict(task=task,language=lang,rule=rule,teams=len(n),median=float(n.median()),minimum=int(n.min()),maximum=int(n.max()),spearman_rho=float(sp.statistic),spearman_p=float(sp.pvalue)))
        for team,count in n.items():teamcounts.append(dict(task=task,language=lang,rule=rule,team=team,submissions=count,test_primary=float(scores[team])))
pd.DataFrame(results).to_csv(O/'frequency_summary.csv',index=False);pd.DataFrame(teamcounts).to_csv(O/'team_counts.csv',index=False)
# These two published correlations are independently reproducible.
assert np.isclose(results[0]['spearman_rho'],.6712455895421081)
en_sub=next(x for x in results if x['task']=='sub' and x['language']=='en');assert np.isclose(en_sub['spearman_rho'],.6501910851472581)
print(pd.DataFrame(results).to_string(index=False))
# Reconstruct the old row-count result independently, with all other inputs fixed.
scores=lb[(lb.language=='ja')&(lb.granularity=='fine')&(lb.metric=='Joint')].set_index('team').accuracy
eligible=r[(r.task=='main')&(r.language=='ja')&r.team.isin(scores.index)&(r.status==3)&r.private_primary.notna()&(r.selected|(r.team=='asukai89'))]
keys=['task','language','team','submission_id']
duplicates=eligible[eligible.duplicated(keys,keep=False)]
legacy_n=eligible.groupby('team').size()
unique_n=eligible.drop_duplicates(keys).groupby('team').size()
legacy=spearmanr(legacy_n,scores.reindex(legacy_n.index))
corrected=spearmanr(unique_n,scores.reindex(unique_n.index))
assert len(duplicates)==2 and duplicates.team.eq('sociocom').all()
assert duplicates.submission_id.nunique()==1 and duplicates.submission_id.iloc[0]=='8830aa88-db2f-4227-94e4-d1b67c497256'
assert legacy_n['sociocom']==2 and unique_n['sociocom']==1
assert abs(legacy.statistic-.4009386883222482)<1e-12 and abs(legacy.pvalue-.22168301553096678)<1e-12
assert abs(corrected.statistic-.4453194528540574)<1e-12 and abs(corrected.pvalue-.16987767013818195)<1e-12
assert (legacy_n.drop('sociocom')==unique_n.drop('sociocom')).all()
audit=dict(status='resolved',cause='same_submission_id_counted_twice_in_legacy_metadata_row_count',language='ja',task='main',team='sociocom',submission_id=duplicates.submission_id.iloc[0],teams=11,legacy_sociocom_count=2,corrected_sociocom_count=1,legacy_rho=float(legacy.statistic),legacy_p=float(legacy.pvalue),corrected_rho=float(corrected.statistic),corrected_p=float(corrected.pvalue),other_team_counts_unchanged=True,performance_scores_unchanged=True)
(O/'duplicate_record_audit.json').write_text(json.dumps(audit,indent=2))
duplicates[['task','language','team','submission_id','datetime','selected','status','public_primary','private_primary']].to_csv(O/'duplicate_metadata_rows.csv',index=False)
print('RESOLVED: legacy rho=.401/P=.222 counts sociocom twice; distinct-ID rho=.445/P=.170 counts it once.')
