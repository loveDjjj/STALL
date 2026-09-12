"""非零起点和乱序/重复请求必须与原随机定位逐像素相同。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2,numpy as np,pytest
from data.video import decode_bounded
from data.sequential_decode import decode_sequential


def test_sequential_pixels_and_request_order(tmp_path):
    path=tmp_path/'clip.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),12,(32,32))
    if not writer.isOpened():pytest.skip('MJPG encoder unavailable')
    rng=np.random.default_rng(17)
    for _ in range(30):writer.write(rng.integers(0,256,(32,32,3),dtype=np.uint8))
    writer.release();indices=[21,7,11,7]
    np.testing.assert_array_equal(decode_sequential(path,indices),decode_bounded(path,indices))


def test_open_failure_uses_original_strict_fallback(monkeypatch):
    import data.sequential_decode as module
    class Closed:
        def isOpened(self):return False
        def release(self):pass
    expected=np.ones((1,2,2,3),dtype=np.uint8)
    monkeypatch.setattr(module,'_open_video_capture',lambda path:Closed())
    monkeypatch.setattr(module,'decode_bounded',lambda path,indices:expected)
    assert module.decode_sequential('missing',[3]) is expected
