"""Reconstruct whole subtask runs from archived matrices; do not pool runs across teams."""
import contextlib,io,re
import numpy as np
import pandas as pd
from common import DATA,OUTPUT,PACKAGE,gold,SubTaskMetric as S
O=OUTPUT/'subtask';O.mkdir(parents=True,exist_ok=True)
rows=[];selected=[]
for lang,ver in [('en',5),('ja',4)]:
    g=gold('sub',lang)[S.COL_NAMES]
    mats={c:pd.read_csv(DATA/f'sources/RadNLP/{c}_{lang}({ver}).csv').rename(columns={'ID':'id'}).set_index(['id','sentence_index']) for c in S.LABEL_COL_NAMES}
    index=pd.MultiIndex.from_frame(g[S.ID_COL_NAMES])
    for c in S.LABEL_COL_NAMES:assert np.array_equal(mats[c].loc[index,'answer'].to_numpy(),g[c].to_numpy())
    for col in mats['measure'].columns:
        if col=='answer':continue
        p=g[S.ID_COL_NAMES].copy()
        for c in S.LABEL_COL_NAMES:p[c]=mats[c].loc[index,col].to_numpy()
        with contextlib.redirect_stdout(io.StringIO()):score=S.compute(g,p)['score']
        team=re.sub(r'-\d+$','',col)
        rows.append(dict(language=lang,team=team,matrix_column=col,**score))
df=pd.DataFrame(rows)
for (lang,team),g in df.groupby(['language','team'],sort=False):
    # idxmax keeps the first historical matrix column in a tie.
    selected.append(g.loc[g.overall_micro_f2.idxmax()].to_dict())
df.to_csv(O/'all_matrix_runs.csv',index=False)
sel=pd.DataFrame(selected);sel['primary_ranking']=~((sel.language=='ja')&(sel.team=='TMUNLPG3'))
sel.to_csv(O/'Tables4_5_subtask_LB.csv',index=False)
print('Subtask primary whole-run scores reconstructed; Japanese TMUNLPG3 metadata is exported separately below')
# TMUNLPG3 Japanese subtask is absent from the historical prediction matrices.
# Preserve the separate platform score, explicitly marked as metadata-only.
import json
registry=pd.read_csv(PACKAGE/'config/submission_registry.csv')
r=registry[(registry.task=='sub')&(registry.language=='ja')&(registry.team=='TMUNLPG3')&registry.selected&(registry.status==3)&registry.private_primary.notna()]
if len(r):
    best=r.sort_values(['private_primary','datetime'],ascending=[False,True]).iloc[0]
    pd.DataFrame([dict(language='ja',team='TMUNLPG3',submission_id=best.submission_id,source='platform_metadata_only_not_vector_recomputed',**json.loads(best.private_scores_json))]).to_csv(O/'Japanese_TMUNLPG3_metadata_scores.csv',index=False)
