"""Reproduce the consensus tumor-size proxy, including its exact T-determination rule."""
import json
import numpy as np
import pandas as pd
from common import DATA,OUTPUT,gold,case_map
O=OUTPUT/'proxy';O.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(DATA/'annotations/main_consensus_proxy.csv');assert len(d)==243 and not d.id.duplicated().any()
for split,n in [('train',108),('val',54),('test',81)]:
    g=gold('main','en',split).set_index('id');a=d[d.partition==split].set_index('id');assert len(a)==n and set(a.index)==set(g.index)
    for c in 'tnm':assert a.loc[g.index,c].equals(g[c])
    assert np.array_equal(case_map().loc[a.index].to_numpy(),a.case_id.to_numpy())
unknown=d.size_unknown.eq(1);high=d.t.isin(['T2a','T2b','T3','T4']);supported=unknown&high
assert ((d.size_t==0)==unknown).all()
assert (d.loc[supported,['extension_t','intrapulmonary_t']].max(axis=1)>=2).all()
summary=[]
for split,sub in [('all',d)]+[(s,d[d.partition==s]) for s in ['train','val','test']]:
    u=sub.size_unknown.eq(1);h=u&sub.t.isin(['T2a','T2b','T3','T4'])
    summary.append(dict(split=split,reports=len(sub),size_indeterminate=int(u.sum()),fraction=float(u.mean()),indeterminate_T2a_or_higher=int(h.sum())))
pd.DataFrame(summary).to_csv(O/'proxy_summary.csv',index=False)
# These are dominant T-determining subtotal groups, not mutually exclusive presence of findings.
a=d.extension_t;b=d.intrapulmonary_t
labels=np.select([supported&(a>b),supported&(b>a),supported&(a==b)&(a>0)],['extension_subtotal_higher','intrapulmonary_subtotal_higher','equal_positive_subtotals'],default='not_in_supported_subset')
d['size_indeterminate']=unknown;d['indeterminate_T2a_or_higher']=supported;d['determining_group']=labels
presence={'extension_only':int((supported&(a>0)&(b==0)).sum()),'intrapulmonary_only':int((supported&(b>0)&(a==0)).sum()),'both_present':int((supported&(a>0)&(b>0)).sum())}
counts=d.loc[supported,'determining_group'].value_counts().to_dict()
assert counts=={'extension_subtotal_higher':20,'intrapulmonary_subtotal_higher':19,'equal_positive_subtotals':1}
assert unknown.sum()==48 and supported.sum()==40
assert d.loc[unknown,'t'].value_counts().to_dict()=={'T4':31,'T0':8,'T3':5,'T2a':4}
assert d.loc[unknown].groupby('partition').size().to_dict()=={'test':26,'train':16,'val':6}
d.to_csv(O/'proxy_report_audit.csv',index=False)
(O/'proxy_verification.json').write_text(json.dumps(dict(status='passed',consensus_reports=243,aligned_gold_labels=True,size_indeterminate=48,T2a_or_higher=40,dominant_subtotal_counts=counts,finding_presence_counts=presence,interpretation='20/19/1 compares T-determining subtotals; it is not the count of mutually exclusive finding presence.',scope='108 train + 54 val + 81 test; additional 135 Japanese-only reports not represented in supplied consensus files'),indent=2))
print(pd.DataFrame(summary).to_string(index=False));print(counts);print('Finding presence, a different classification:',presence)
