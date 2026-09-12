"""从官方下载包恢复指定短视频，按成员CRC与SHA256核验，不覆盖不同内容。"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from artifacts import paper_json


def recover(root,kind):
    root=Path(root);base=root/'datasets/recovery'
    if kind=='genvideo':
        archive=base/'archives/GenVideo-Val.zip'
        wanted=None
    else:
        archive=base/'archives'/f'videofeedback_{kind}.zip'
        metadata=json.loads((base/'metadata'/f'videofeedback_{kind}.json').read_text())
        wanted={str(r['id']) for r in metadata if 'vidprom_hotshot/' in r['video link'] and r['dynamic degree']>=3}
    records=[]
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path=Path(member.filename)
            if path.suffix.lower()!='.mp4':continue
            if kind=='genvideo':
                if path.parent.name not in ('HotShot','MoonValley'):continue
                destination=root/'datasets/genvideo/fake'/path.parent.name/path.name
            else:
                if path.stem not in wanted:continue
                destination=root/'datasets/videofeedback/fake/Hotshot-XL'/path.name
            blob=z.read(member)  # ZipFile验证成员CRC。
            digest=hashlib.sha256(blob).hexdigest()
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'已有视频内容不同：{destination}')
            else:
                temporary=destination.with_suffix('.recovering')
                temporary.write_bytes(blob);temporary.replace(destination)
            records.append(dict(path=str(destination.relative_to(root)),member=member.filename,
                bytes=len(blob),sha256=digest,crc=member.CRC))
    if not records:raise ValueError('归档未找到目标视频')
    output=base/'receipts'/f'{kind}.json'
    paper_json(output,dict(status='completed',archive=str(archive),videos=records))
    print(f'{kind}: verified {len(records)} videos',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['genvideo','train','test'])
    args=parser.parse_args();recover(Path.cwd(),args.kind)
