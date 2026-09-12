"""按原顺序消费的有界CPU预取；不导入研究脚本，不在工作线程调用GPU。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2

from data.video import _open_video_capture


def bounded_map(items, prepare, estimate, *, workers=4, depth=4, budget=4096 * 2**20):
    """内存准入计入当前消费项；退出/异常时等待在途线程并取消未开始任务。"""
    if any(type(x) is not int or x < 1 for x in (workers, depth, budget)):
        raise ValueError("线程、预取深度和预算须为正整数")
    iterator = iter(items)
    pending = deque()
    used = 0
    waiting = None
    exhausted = False
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paper-decode")
    try:
        while True:
            while len(pending) < depth and not exhausted:
                if waiting is None:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    size = estimate(item)
                    if not isinstance(size, int) or size < 1 or size > budget:
                        raise ValueError("单视频预留超过预算或估算无效")
                    waiting = item, size
                item, size = waiting
                if used + size > budget:
                    break
                pending.append((item, size, pool.submit(prepare, item)))
                used += size
                waiting = None
            if not pending:
                break
            item, size, future = pending.popleft()
            result = future.result()
            yield item, result
            del result, future
            used -= size
    finally:
        for _, _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def frame_reservation(path, count):
    """估算BGR解码、float32预处理及stack峰值；不是整个进程RSS硬上限。"""
    if type(count) is not int or count < 1:
        raise ValueError("需要正整数帧数")
    cap = _open_video_capture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if width < 1 or height < 1:
        raise ValueError(f"无法读取预取视频尺寸：{path}")
    return (2 * count + 18) * width * height * 3 + 2 * count * 3 * 224 * 224 * 4


def staged_map(
    items, prepare, advance, finish, estimate, *, workers=4, depth=4, budget=4096 * 2**20
):
    """CPU粗准备→消费线程选窗→CPU密集准备，最多提前推进一个视频的选窗。"""
    if any(type(x) is not int or x < 1 for x in (workers, depth, budget)):
        raise ValueError("线程、预取深度和预算须为正整数")
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paper-score-decode")
    iterator = iter(items)
    pending = deque()
    waiting = None
    used = 0
    exhausted = False

    def promote(entry):
        if entry["coarse"] is not None:
            prepared = entry["coarse"].result()
            entry["plan"] = advance(entry["item"], prepared)
            entry["coarse"] = None
            del prepared
            entry["dense"] = pool.submit(finish, entry["item"], entry["plan"])

    try:
        while True:
            while len(pending) < depth and not exhausted:
                if waiting is None:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    size = estimate(item)
                    if type(size) is not int or size < 1 or size > budget:
                        raise ValueError("单视频预留超过预算或估算无效")
                    waiting = item, size
                item, size = waiting
                if used + size > budget:
                    break
                pending.append(
                    dict(
                        item=item,
                        size=size,
                        coarse=pool.submit(prepare, item),
                        dense=None,
                        plan=None,
                    )
                )
                used += size
                waiting = None
            if not pending:
                break
            entry = pending[0]
            promote(entry)
            if len(pending) > 1:
                following = pending[1]
                if following["coarse"] is not None and following["coarse"].done():
                    promote(following)
                del following
            frames = entry["dense"].result()
            yield entry["item"], entry["plan"], frames
            used -= entry["size"]
            pending.popleft()
            del frames, entry
    finally:
        for entry in pending:
            for phase in ("coarse", "dense"):
                if entry[phase] is not None:
                    entry[phase].cancel()
        pool.shutdown(wait=True, cancel_futures=True)
