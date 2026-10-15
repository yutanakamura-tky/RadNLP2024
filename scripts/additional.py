"""Paired mean CIs, deterministic subtask baseline, label variation and cohort audit."""
import contextlib,io,json
import numpy as np
import pandas as pd
from scipy.stats import t
from common import OUTPUT,PACKAGE,gold,case_map,SubTaskMetric as S
O=OUTPUT/'additional';O.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(OUTPUT/'cross_language/paired_case_results.csv')
draw=np.random.default_rng(42).integers(0,9,size=(10000,9));rows=[]
for cat,g in D.groupby('category',sort=False):
    x=g.sort_values('case_id').difference_numerator.to_numpy()/27*100
    lo,hi=np.quantile(x[draw].mean(axis=1),[.025,.975]);se=x.std(ddof=1)/np.sqrt(9);crit=t.ppf(.975,8)
    rows.append(dict(category=cat,mean_difference_pp=x.mean(),paired_bootstrap_low_pp=lo,paired_bootstrap_high_pp=hi,paired_t_low_pp=x.mean()-crit*se,paired_t_high_pp=x.mean()+crit*se,cases=9,draws=10000,seed=42))
pd.DataFrame(rows).to_csv(O/'Table7_paired_mean_difference_CI.csv',index=False)
summary=[];labels=[];variation=[];percase=[];dataset=[];proxy=[];mapping=case_map()
for lang in ['en','ja']:
    g=gold('sub',lang)[S.COL_NAMES];p=g.copy();p[S.LABEL_COL_NAMES]=0;p.omittable=1
    with contextlib.redirect_stdout(io.StringIO()):scores=S._compute(g,p)
    tiny=g.iloc[:1].copy();tiny[S.LABEL_COL_NAMES]=0;tiny.measure=1;bad=tiny.copy();bad.omittable=1
    with contextlib.redirect_stdout(io.StringIO()):a=S._compute(tiny,tiny.copy());b=S._compute(tiny,bad)
    assert a['overall_micro_f2']==b['overall_micro_f2']==1 and a['inclusion_micro_f2']==1 and b['inclusion_micro_f2']==0
    matched=g[g.id.isin(gold('main','en').id)]
    summary.append(dict(language=lang,total_sentences=len(g),reports=g.id.nunique(),omittable_count=int(g.omittable.sum()),all_staging_negative=int(g[S.LABEL_COL_NAMES[1:]].eq(0).all(axis=1).sum()),contradictory_reference=int(((g.omittable==1)&g[S.LABEL_COL_NAMES[1:]].eq(1).any(axis=1)).sum()),**scores,matched_reports=matched.id.nunique(),matched_sentences=len(matched),matched_sentences_per_report=len(matched)/matched.id.nunique()))
    main=gold('main',lang);main['case_id']=mapping.loc[main.id].to_numpy()
    for c in 'tnm':
        counts=main.groupby('case_id')[c].nunique()
        variation.append(dict(language=lang,category=c.upper(),n_cases=len(counts),all_9_same_gold_cases=int((counts==1).sum()),varying_gold_cases=int((counts>1).sum()),varying_gold_fraction=float((counts>1).mean()),max_unique_gold_labels_within_case=int(counts.max())))
        for case,n in counts.items():percase.append(dict(language=lang,category=c.upper(),case_id=case,unique_reference_labels=n))
    for split in ['train','val','test']:
        m=gold('main',lang,split);s=gold('sub',lang,split)
        dataset.append(dict(language=lang,split=split,reports=len(m),imaging_cases=mapping.loc[m.id].nunique(),subtask_sentences=len(s)))
        for c in 'tnm':
            for value,n in m[c].value_counts().items():labels.append(dict(task='main',language=lang,split=split,category=c,label=value,count=n))
        for c in S.LABEL_COL_NAMES:labels.append(dict(task='sub',language=lang,split=split,category=c,label='positive',count=int(s[c].sum())))
        en=gold('main','en',split);ja=gold('main','ja',split);common=sorted(set(en.id)&set(ja.id))
        assert en.set_index('id').loc[common,list('tnm')].equals(ja.set_index('id').loc[common,list('tnm')])
        if lang=='ja':
            for cohort,subset in [('language_aligned',ja[ja.id.isin(common)]),('full_Japanese',ja)]:
                for label,n in subset.t.value_counts().items():proxy.append(dict(split=split,cohort=cohort,label=label,count=n))
for name,rows in [('Subtask_baseline_and_matched_sentence_counts',summary),('caselevel_gold_variation',variation),('S9_reference_variation_by_case',percase),('Table1_dataset_counts',dataset),('S2_label_counts',labels),('proxy_cohort_T_counts',proxy)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
print('Table 7 CIs, baseline, matched sentences and gold-label audits completed')
