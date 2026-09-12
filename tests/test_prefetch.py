"""主线CPU预取的顺序、内存准入、退出和异常合同。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.prefetch import bounded_map
from data.prefetch import staged_map
import threading


def test_order_and_budget_include_consumer():
    assert list(
        bounded_map(range(19), lambda x: x * 2, lambda x: 1, workers=3, depth=8, budget=5)
    ) == [(i, i * 2) for i in range(19)]
    submitted = []

    def prepare(i):
        submitted.append(i)
        return i

    stream = bounded_map(range(4), prepare, lambda x: 2, workers=1, depth=8, budget=2)
    assert next(stream) == (0, 0) and submitted == [0]
    assert next(stream) == (1, 1)
    stream.close()


def test_failure_and_invalid_reservation():
    with pytest.raises(ValueError, match="预算"):
        list(bounded_map([1], str, lambda x: 5, budget=4))
    with pytest.raises(ValueError):
        list(bounded_map([1], str, lambda x: 1, workers=0))

    def fail(x):
        raise RuntimeError("解码失败")

    with pytest.raises(RuntimeError, match="解码失败"):
        list(bounded_map([1, 2], fail, lambda x: 1))
    assert list(bounded_map([], str, lambda x: 1)) == []


def test_staged_thread_roles_order_and_budget():
    owner = threading.get_ident()
    submitted = []

    def prepare(i):
        assert threading.get_ident() != owner
        submitted.append(i)
        return i * 2

    def advance(i, value):
        assert threading.get_ident() == owner
        return value + 1

    def finish(i, plan):
        assert threading.get_ident() != owner
        return plan * 3

    stream = staged_map(
        range(4), prepare, advance, finish, lambda i: 1, workers=2, depth=4, budget=1
    )
    assert next(stream) == (0, 1, 3) and submitted == [0]
    assert next(stream) == (1, 3, 9)
    stream.close()
    actual = list(
        staged_map(range(10), prepare, advance, finish, lambda i: 1, workers=3, depth=4, budget=4)
    )
    assert actual == [(i, i * 2 + 1, (i * 2 + 1) * 3) for i in range(10)]


@pytest.mark.parametrize("phase", ["prepare", "advance", "finish"])
def test_staged_failure_propagates_and_drains(phase):
    def first(i):
        if phase == "prepare":
            raise RuntimeError("phase failure")
        return i

    def second(i, value):
        if phase == "advance":
            raise RuntimeError("phase failure")
        return value

    def third(i, value):
        if phase == "finish":
            raise RuntimeError("phase failure")
        return value

    with pytest.raises(RuntimeError, match="phase failure"):
        list(staged_map([1, 2], first, second, third, lambda i: 1))
