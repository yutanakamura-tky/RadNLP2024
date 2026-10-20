"""Compare fresh outputs with frozen final-analysis snapshots, never with copied outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import OUTPUT,PACKAGE
checks=[]
a=json.loads((OUTPUT/'scoring/scoring_behavior_checks.json').read_text());e=json.loads((PACKAGE/'expected/scoring_behavior_checks.json').read_text());assert a==e
checks.append(dict(output='scoring/scoring_behavior_checks.json',checks=5,status='passed'))
def compare(actual,expected,sort=None):
    a=pd.read_csv(OUTPUT/actual);e=pd.read_csv(PACKAGE/'expected'/expected)
    assert set(a.columns)==set(e.columns),(actual,'columns')
    a=a[e.columns]
    keys=sort or list(e.columns)
    a=a.sort_values(keys,na_position='last').reset_index(drop=True);e=e.sort_values(keys,na_position='last').reset_index(drop=True)
    pd.testing.assert_frame_equal(a,e,check_exact=False,rtol=1e-12,atol=1e-12,check_dtype=False)
    checks.append(dict(output=actual,rows=len(a),columns=len(a.columns),status='passed',tolerance=1e-12))
compare('selection/representative_runs.csv','representative_runs.csv',['language','team'])
for folder in ['leaderboard','cross_language','ensemble','supplementary']:
    for f in sorted((PACKAGE/'expected'/folder).glob('*.csv')):compare(folder+'/'+f.name,folder+'/'+f.name)
for f in ['Table7_paired_mean_difference_CI.csv','Subtask_baseline_and_matched_sentence_counts.csv','caselevel_gold_variation.csv']:compare('additional/'+f,f)
compare('confusion/confusion_counts_percentages.csv','confusion_counts_percentages.csv',['language','category','reference','prediction'])
a=pd.read_csv(OUTPUT/'subtask/Tables4_5_subtask_LB.csv');e=pd.read_csv(PACKAGE/'expected/Tables4_5_displayed_scores.csv')
for r in e.itertuples(index=False):
    found=a[(a.language==r.language)&(a.team==r.team)];assert len(found)==1
    for col in e.columns[2:]:assert abs(found.iloc[0][col]-getattr(r,col))<=0.00005000001,(r.language,r.team,col)
checks.append(dict(output='subtask/Tables4_5_subtask_LB.csv',rows=len(e),status='passed',tolerance='displayed 4-decimal rounding'))
# Cohort audit is separate from the detailed consensus-based proxy check below.
p=pd.read_csv(OUTPUT/'additional/proxy_cohort_T_counts.csv');q=p[p.cohort=='language_aligned'].groupby('label')['count'].sum()
assert {x:int(q[x]) for x in ['T0','T2a','T3','T4']}=={'T0':8,'T2a':36,'T3':22,'T4':93}
assert q.sum()==243
checks.append(dict(output='additional/proxy_cohort_T_counts.csv',status='passed',scope='cohort-only check; detailed size counts verified separately below'))
audit=json.loads((OUTPUT/'submission_frequency/duplicate_record_audit.json').read_text())
assert audit['status']=='resolved' and audit['legacy_sociocom_count']==2 and audit['corrected_sociocom_count']==1
assert abs(audit['legacy_rho']-.4009386883222482)<1e-12 and abs(audit['legacy_p']-.22168301553096678)<1e-12
assert abs(audit['corrected_rho']-.4453194528540574)<1e-12 and abs(audit['corrected_p']-.16987767013818195)<1e-12
checks.append(dict(output='submission_frequency/duplicate_record_audit.json',status='passed',scope='legacy row-count result and corrected distinct-ID result both independently reconstructed'))
proxy=json.loads((OUTPUT/'proxy/proxy_verification.json').read_text())
assert proxy['status']=='passed' and proxy['size_indeterminate']==48 and proxy['T2a_or_higher']==40
assert proxy['dominant_subtotal_counts']=={'extension_subtotal_higher':20,'intrapulmonary_subtotal_higher':19,'equal_positive_subtotals':1}
checks.append(dict(output='proxy/proxy_verification.json',status='passed',scope='consensus-derived 48/243, split counts, 40/48 and dominant T-subtotal groups'))
(OUTPUT/'verification.json').write_text(json.dumps(dict(status='passed',comparisons=checks),indent=2))
print(f'PASS: {len(checks)} output comparisons and cohort checks')
