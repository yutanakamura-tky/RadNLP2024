"""Re-evaluate all available eligible variants and select representatives without test scores."""
import hashlib,contextlib,io
import numpy as np
import pandas as pd
from common import DATA,OUTPUT,PACKAGE,gold,MainTaskMetric as M
O=OUTPUT/'selection';O.mkdir(parents=True,exist_ok=True)
files=pd.read_csv(PACKAGE/'config/prediction_files.csv')
inv=pd.read_csv(PACKAGE/'config/representative_candidates.csv')
checks=[]
for f in files.itertuples():
    raw=pd.read_csv(DATA/f.prediction_file)
    assert not raw.id.duplicated().any(),f.prediction_file
    d=dict(language=f.language,team=f.team,submission_id=f.submission_id,prediction_file=f.prediction_file,sha256=hashlib.sha256((DATA/f.prediction_file).read_bytes()).hexdigest())
    assert d['sha256']==f.sha256
    for split in ['val','test']:
        g=gold('main',f.language,split)[M.COL_NAMES]
        d[split+'_coverage']=len(set(g.id)&set(raw.id))
        if d[split+'_coverage']:
            with contextlib.redirect_stdout(io.StringIO()):score=M.compute(g,raw)['score']
            d[split+'_joint_fine']=score['joint_accuracy_fine']
    checks.append(d)
checks=pd.DataFrame(checks);checks.to_csv(O/'candidate_scores.csv',index=False)
rows=[]
for (lang,team),cand in inv.groupby(['language','team']):
    observed=[]
    for c in cand.itertuples():
        variants=checks[(checks.language==lang)&(checks.team==team)&(checks.submission_id==c.submission_id)]
        assert c.selected or team=='asukai89'
        for f in variants[variants.test_coverage>0].to_dict('records'):
            observed.append(dict(f,datetime=c.datetime,selected=c.selected,platform_team_id=c.platform_team_id))
    measured=[x for x in observed if x['val_coverage']>0]
    if measured:
        pick=sorted(measured,key=lambda x:(-x['val_joint_fine'],x['datetime'],x['prediction_file']))[0];rule='1'
        for c in cand.itertuples():
            if not ((checks.language==lang)&(checks.team==team)&(checks.submission_id==c.submission_id)).any():
                upper=max(float(x) for x in str(c.metadata_validation_values).split(';'))
                assert upper<=pick['val_joint_fine']+1e-12
                if abs(upper-pick['val_joint_fine'])<1e-12:assert c.datetime>=pick['datetime']
    else:
        assert observed,(lang,team)
        assert cand.raw_variant_count.gt(0).all()
        pick=sorted(observed,key=lambda x:(x['datetime'],x['prediction_file']))[0];rule='2'
    expected=len(gold('main',lang));assert pick['test_coverage']==expected
    rows.append(dict(language=lang,team=team,status='confirmed_with_selected_exception' if team=='asukai89' else 'confirmed',rule=rule,platform_team_id=pick['platform_team_id'],submission_id=pick['submission_id'],datetime=pick['datetime'],selected=pick['selected'],validation_joint_fine=pick.get('val_joint_fine'),validation_coverage=pick['val_coverage'],test_coverage=pick['test_coverage'],expected_test_reports=expected,prediction_file=pick['prediction_file'],sha256=pick['sha256']))
result=pd.DataFrame(rows);assert len(result)==21
result.to_csv(O/'representative_runs.csv',index=False)
# Measured zero is distinct from missing validation predictions.
assert set(result[result.rule=='2'].team)=={'CYUT','ATILF','NLI24','UoM'}
print('Representative selection confirmed for 21 team-language pairs')
