import json
from pathlib import Path
import pandas as pd
from reference import file_digest
from artifacts import paper_json
from looped_video import evaluation


def test_partial_report_does_not_invent_three_domain_average(tmp_path,monkeypatch):
    out=tmp_path/'run';out.mkdir();c=dict(run_directory='run',seeds=[17],variants=['looped','untied'])
    monkeypatch.setattr(evaluation,'configuration',lambda root:c)
    tasks=[dict(key=f'comgenvid__{v}__s17',fold='comgenvid',variant=v,seed=17) for v in c['variants']]
    paper_json(out/'tasks.json',dict(tasks=tasks));task=tasks[0]
    frame=pd.DataFrame(dict(video_id=['a','b','c','d'],dataset=['comgenvid']*4,subset=['real','real','annotated','annotated'],
        source_model=['real','real','fake','fake'],source_group=['a','b','c','d'],split_group=['a','b','c','d'],
        final_score=[2.,1.,-1.,-2.],fake_logit=[-2.,-1.,1.,2.],p_fake=[.1,.2,.8,.9],variant=['looped']*4,seed=[17]*4,fold=['comgenvid']*4))
    d=out/'evaluation'/task['key'];d.mkdir(parents=True);frame.to_csv(d/'scores.csv',index=False)
    paper_json(d/'manifest.json',dict(scores_sha256=file_digest(d/'scores.csv'),seconds=None))
    t=out/'training'/task['key'];t.mkdir(parents=True);paper_json(t/'manifest.json',dict(seconds=5.,parameters=100))
    frame[['video_id','dataset','subset']].assign(generator='fake').to_csv(out/'pairs.csv',index=False)
    evaluation.report(tmp_path,None)
    status=json.loads((out/'status.json').read_text());assert status['status']=='partial_results'
    assert status['models_evaluated']==1 and len(status['missing'])==1
    assert pd.read_csv(out/'macro_metrics.csv').empty and not (out/'average_summary.csv').exists()
    assert (out/'RESULTS_zh.md').exists()
