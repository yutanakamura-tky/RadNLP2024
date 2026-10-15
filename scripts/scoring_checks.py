"""Reproduce the audit's controlled checks of official scoring behavior."""
import contextlib,io,json
import pandas as pd
from common import OUTPUT,MainTaskMetric as M
O=OUTPUT/'scoring';O.mkdir(parents=True,exist_ok=True)
g=pd.DataFrame({'id':[1,2],'t':['T2a','T4'],'n':['N0','N1'],'m':['M0','M1a']})
checks=[]
for name,p in [
    ('format_normalization',pd.DataFrame({'id':[1,2],'t':[' 2A ','t4'],'n':['0','n1'],'m':['0','m1A']})),
    ('invalid_coarse_labels',pd.DataFrame({'id':[1,2],'t':['T2','T4'],'n':['N0','N1'],'m':['M0','M1']})),
    ('missing_report',g.iloc[:1]),('missing_T_column',g.drop(columns='t')),
    ('duplicate_id',pd.concat([g,g.iloc[:1]],ignore_index=True))]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):result=M.compute(g.copy(),p.copy())['score']
        checks.append(dict(check=name,result=result))
    except Exception as e:checks.append(dict(check=name,error=type(e).__name__+': '+str(e)))
(O/'scoring_behavior_checks.json').write_text(json.dumps(checks,indent=2))
print('Five controlled official-scoring behavior checks completed')
