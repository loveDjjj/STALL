"""新增短视频三个单元的明确1秒身份和共享真实配对。"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import paper_json
from reference import file_digest


def one_second(row):
    row=dict(row);indices=json.loads(row['downsample_idxs'])
    if len(indices)<8:raise ValueError('不足8个有效采样帧')
    first=np.random.RandomState(42).randint(0,len(indices)-8+1)
    row['full_downsample_idxs']=row['downsample_idxs']
    row['downsample_idxs']=json.dumps(indices[first:first+8])
    row['1_sec_idxs']=row['downsample_idxs'];row['2_sec_idxs']=''
    row['original_video_id']=row['video_id'];row['video_id']+=':duration1'
    row['evaluation_duration']=1
    return row


def prepare(root,domain):
    root=Path(root);active=pd.read_csv(root/'data/manifests/active'/domain/'evaluation.csv',keep_default_na=False)
    fields=active.columns.tolist();candidates=[]
    if domain=='genvideo':
        old=pd.read_csv(root/'cache/contracts/data/manifests/development/genvideo_evaluation.csv',keep_default_na=False)
        for _,r in old[old.source_model.isin(['HotShot','MoonValley'])].iterrows():
            path=root/r.video_path
            if not path.is_file():raise FileNotFoundError(path)
            row={k:'' for k in fields};row.update(r.to_dict())
            vid=f'genvideo:{r.video_path}'
            row.update(dataset=domain,split='evaluation',video_id=vid,legacy_video_id=vid,
                legacy_path=r.video_path,source_group=vid,generator=r.source_model)
            candidates.append(one_second(row))
    else:
        metadata={}
        for s in ('train','test'):
            for r in json.loads((root/'datasets/recovery/metadata'/f'videofeedback_{s}.json').read_text()):
                if 'vidprom_hotshot/' in r['video link'] and r['dynamic degree']>=3:metadata[str(r['id'])]=r
        for key,m in sorted(metadata.items()):
            path=root/'datasets/videofeedback/fake/Hotshot-XL'/f'{key}.mp4'
            if not path.is_file():raise FileNotFoundError(path)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=avg_frame_rate,nb_frames,duration','-of','json',str(path)]))['streams'][0]
            a,b=map(float,probe['avg_frame_rate'].split('/'));fps=a/b
            count=int(probe['nb_frames']);duration=float(probe['duration'])
            if fps<8 or count<8:continue
            indices=[];j=0
            while round(fps*j/8)<count:
                indices.append(round(fps*j/8));j+=1
            if len(indices)<8:continue
            relative=str(path.relative_to(root));vid=f'{domain}:{relative}'
            row={k:'' for k in fields};row.update(video_path=relative,subset='annotated',source_model='Hotshot-XL',
                fps=fps,num_frames=count,duration_seconds=duration,downsample_idxs=json.dumps(indices),
                dataset=domain,split='evaluation',video_id=vid,legacy_video_id=vid,legacy_path=relative,
                source_group=vid,generator='Hotshot-XL',dynamic_degree=m['dynamic degree'])
            candidates.append(one_second(row))
    fake=pd.DataFrame(candidates)
    real=pd.DataFrame([one_second(row) for row in active[active.subset=='real'].to_dict('records')])
    pairs=[]
    for generator,g in fake.groupby('source_model'):
        count=min(len(g),len(real));sources=real.source_model.nunique();count=count//sources*sources
        picked_fake=g.sample(n=count,random_state=42)
        picked_real=pd.concat([r.sample(n=count//sources,random_state=42) for _,r in real.groupby('source_model')])
        for row in pd.concat([picked_real,picked_fake]).to_dict('records'):
            pairs.append({k:row[k] for k in ('video_id','dataset','subset')}|{'generator':generator})
    # 只评分正式配对涉及的唯一身份；fake余量清单另外保存，不按得分挑样本。
    pairs=pd.DataFrame(pairs);frame=pd.concat([real,fake],ignore_index=True)
    frame=frame[frame.video_id.isin(set(pairs.video_id))].fillna('')
    out=root/'data/manifests/short_video'/domain;out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'evaluation.csv',index=False);pairs.to_csv(out/'pairs.csv',index=False)
    fake.to_csv(out/'eligible_fake_pool.csv',index=False)
    paper_json(out/'manifest.json',dict(status='prepared',rule='8fps 1sec official seed42 first draw; real/fake same frames and pool; balanced seed42',
        unique_videos=len(frame),pair_rows=len(pairs),generators=sorted(fake.source_model.unique()),
        source_evaluation_sha256=file_digest(root/'data/manifests/active'/domain/'evaluation.csv'),
        files={p.name:file_digest(p) for p in out.glob('*.csv')}))
    print(domain,len(frame),len(pairs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('domain',choices=['genvideo','videofeedback'])
    prepare(Path.cwd(),p.parse_args().domain)
