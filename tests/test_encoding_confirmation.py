"""H264处理必须保持奇数图像边长、帧数和相对时间戳，不能偷偷补帧。"""
import shutil,subprocess
import numpy as np
import pytest
from evaluation.encoding_confirmation import timestamps,transcode


@pytest.mark.skipif(shutil.which('ffmpeg') is None,reason='ffmpeg unavailable')
def test_transcode_keeps_odd_dimensions_and_timing(tmp_path):
    source=tmp_path/'source.mkv';target=tmp_path/'encoded.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc=size=65x49:rate=8:duration=1','-c:v','ffv1',str(source)],check=True)
    c=dict(codec='libx264',preset='medium',pixel_format='yuv444p',encoder_threads=1)
    transcode(source,target,23,c)
    a=timestamps(source);b=timestamps(target)
    assert len(a)==len(b)==8
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-3)
    text=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','csv=p=0',str(target)],check=True,capture_output=True,text=True).stdout.strip()
    assert text=='65,49'
