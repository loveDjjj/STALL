"""同帧索引顺序解码，减少重复关键帧解码；失败回退原严格路径。"""
from pathlib import Path
import cv2
import numpy as np
from data.video import _open_video_capture,decode_bounded
from selection import validate_indices,uniform_windows


def decode_sequential(path,frame_indices):
    requested=list(frame_indices);indices=validate_indices(sorted(set(requested)))
    if not indices:raise ValueError('解码索引为空')
    capture=_open_video_capture(path);found={};wanted=set(indices)
    try:
        if capture.isOpened():
            if indices[0]:capture.set(cv2.CAP_PROP_POS_FRAMES,indices[0])
            for index in range(indices[0],indices[-1]+1):
                ok,frame=capture.read()
                if not ok:break
                if index in wanted:found[index]=frame
    finally:capture.release()
    if wanted-set(found):return decode_bounded(path,requested)
    return np.stack([found[index] for index in requested])


class SequentialDecodeMixin:
    """保留原GPU批次边界，粗扫/Uniform参考共用一次CPU解码。"""
    def prepare_coarse(self,path,indices,*,selector='feature_change',k=3,include_uniform=False,materialize=True):
        indices=validate_indices(indices)
        if len(indices)<16 or k not in (1,2,3) or selector not in ('uniform','feature_change'):raise ValueError('观察参数错误')
        coarse=indices[::8] if selector=='feature_change' else []
        uniform=uniform_windows(indices,1)[0] if include_uniform else None
        requested=sorted(set(coarse)|set(uniform.frame_indices if uniform else []))
        if requested:
            prepared=self.extractor.prepare_frames(decode_sequential(Path(path),requested))
            positions={v:i for i,v in enumerate(requested)}
            uniform_pair=(uniform,prepared[[positions[i] for i in uniform.frame_indices]]) if uniform else None
            coarse_frames=prepared[[positions[i] for i in coarse]] if coarse else []
        else:uniform_pair=None;coarse_frames=[]
        return dict(uniform=uniform_pair,coarse=[coarse_frames[i:i+8] for i in range(0,len(coarse),8)],coarse_count=len(coarse))

    def prepare_dense(self,path,plan):
        return self.extractor.prepare_frames(decode_sequential(Path(path),plan['union']))
