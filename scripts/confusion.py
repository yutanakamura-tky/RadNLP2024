"""Re-select historical LB matrix columns and regenerate Figures 7 and 8."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import DATA,OUTPUT,PACKAGE,gold,MainTaskMetric as M
O=OUTPUT/'confusion';O.mkdir(parents=True,exist_ok=True)
allowed={'t':['T0','Tis','T1mi','T1a','T1b','T1c','T2a','T2b','T3','T4'],'n':['N0','N1','N2','N3'],'m':['M0','M1a','M1b','M1c']}
def teamname(col):
    if col.startswith('TMUNLPG3-'):return 'TMUNLPG3'
    if col.startswith('Ubie'):return 'Ubie'
    return {'NITKC team':'NITKC'}.get(col,col)
manifest=pd.read_csv(PACKAGE/'config/leaderboard_runs.csv');cells=[];invalid=[];chosen_rows=[]
for lang,name,ver,nteams,figid in [('en','English',3,10,7),('ja','Japanese',2,11,8)]:
    g=gold('main',lang)[M.COL_NAMES];pdir=DATA/'sources/RadNLP2024'
    mats={c:pd.read_csv(pdir/f'{c}_{lang}({ver}).csv').set_index('ID') for c in 'tnm'}
    for c in 'tnm':assert mats[c].loc[g.id,'answer'].tolist()==g[c].tolist()
    preds={col:M.preprocess_submission(g,pd.DataFrame({'id':g.id,**{c:mats[c].loc[g.id,col].to_numpy() for c in 'tnm'}})) for col in mats['t'].columns if col!='answer'}
    chosen={team:max([c for c in preds if teamname(c)==team],key=lambda c:(preds[c][list('tnm')].to_numpy()==g[list('tnm')].to_numpy()).all(axis=1).mean()) for team in sorted({teamname(c) for c in preds})}
    assert len(chosen)==nteams
    for team,col in chosen.items():
        expected=manifest[(manifest.language==lang)&(manifest.team==team)].iloc[0]
        assert col==expected.matrix_column
        chosen_rows.append(dict(language=lang,team=team,matrix_column=col,raw_submission_identity='unresolved' if pd.isna(expected.exact_submission_matches) else 'one or more vector matches; not unique identity'))
    plotdata={}
    for c in 'tnm':
        labels=allowed[c];refs=[s for s in labels if g[c].eq(s).any()];predlabels=labels+['Other/invalid']
        counts=np.zeros((len(refs),len(predlabels)),dtype=int)
        for team,col in chosen.items():
            p=preds[col]
            for y,v in zip(g[c],p[c]):counts[refs.index(y),predlabels.index(v) if v in labels else -1]+=1
            for value,n in p.loc[~p[c].isin(labels),c].value_counts(dropna=False).items():invalid.append(dict(language=lang,team=team,category=c.upper(),label=str(value),count=int(n)))
        n=np.array([g[c].eq(s).sum() for s in refs]);assert np.array_equal(counts.sum(axis=1),n*nteams)
        pct=counts/counts.sum(axis=1)[:,None]*100;assert np.allclose(pct.sum(axis=1),100)
        plotdata[c]=(refs,predlabels,n,pct)
        for i,y in enumerate(refs):
            for j,v in enumerate(predlabels):cells.append(dict(language=lang,category=c.upper(),reference=y,prediction=v,count=int(counts[i,j]),row_denominator=int(counts[i].sum()),percent=float(pct[i,j]),unique_reference_reports=int(n[i]),teams=nteams))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(12,10.2));gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],hspace=.52,wspace=.36)
    axes=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])];fig.subplots_adjust(left=.12,right=.86,bottom=.20,top=.88)
    for ax,c in zip(axes,'tnm'):
        refs,predlabels,n,pc=plotdata[c];im=ax.imshow(pc,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(predlabels)),[v if v!='Other/invalid' else 'Other/\ninvalid' for v in predlabels],rotation=40,ha='right');ax.set_yticks(range(len(refs)),[f'{v} (n={nn})' for v,nn in zip(refs,n)])
        ax.set_title(c.upper()+' classification',pad=10);ax.set_xlabel('Prediction');ax.set_ylabel('Reference')
        ax.set_xticks(np.arange(-.5,len(predlabels),1),minor=True);ax.set_yticks(np.arange(-.5,len(refs),1),minor=True);ax.grid(which='minor',color='white',linewidth=.45,alpha=.55);ax.tick_params(which='both',length=0)
        for i in range(len(refs)):
            for j in range(len(predlabels)):ax.text(j,i,f'{pc[i,j]:.1f}',ha='center',va='center',fontsize=9,color='black' if pc[i,j]>50 else 'white')
    cb=fig.colorbar(im,cax=fig.add_axes([.9,.21,.019,.52]));cb.set_label('Row-normalized prediction rate (%)')
    fig.suptitle(f'{name} main task: confusion matrices\nOne leaderboard run per team ({nteams} teams)',fontsize=15,y=.97)
    fig.text(.12,.025,f'n = unique reference reports; each contributes {nteams} team predictions.\nOther/invalid predictions are retained; each row sums to 100% before rounding.',fontsize=9)
    fig.savefig(O/f'Figure{figid}_{lang}_LB_confusion.png',dpi=600);fig.savefig(O/f'Figure{figid}_{lang}_LB_confusion.pdf');fig.savefig(O/f'preview_{lang}.png',dpi=130);plt.close(fig)
pd.DataFrame(cells).to_csv(O/'confusion_counts_percentages.csv',index=False)
pd.DataFrame(invalid).to_csv(O/'other_invalid_predictions.csv',index=False)
pd.DataFrame(chosen_rows).to_csv(O/'selected_matrix_columns.csv',index=False)
print('LB matrix selection and all confusion denominators verified')
