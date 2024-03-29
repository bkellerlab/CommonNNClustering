import pytest

from commonnn import _types


@pytest.mark.parametrize(
    "queue_type",
    [
        (_types.QueueFIFODeque),
        (_types.QueueExtFIFOQueue),
        (_types.QueueExtLIFOVector),
        (_types.PriorityQueueMaxHeap),
        (_types.PriorityQueueExtMaxHeap)
    ]
)
def test_init_queue(queue_type, file_regression):
    q = queue_type()
    file_regression.check(f"{q!r}\n{q!s}")


@pytest.mark.parametrize(
    "queue_type,kind",
    [
        (_types.QueueFIFODeque, "fifo"),
        (_types.QueueExtFIFOQueue, "fifo"),
        (_types.QueueExtLIFOVector, "lifo"),
    ]
)
def test_use_queue(queue_type, kind):
    queue = queue_type()
    assert queue.is_empty()

    queue.push(1)
    assert not queue.is_empty()
    assert queue.size() == 1
    assert queue.pop() == 1

    pushed = list(range(10))
    for i in pushed:
        queue.push(i)

    popped = []
    while not queue.is_empty():
        popped.append(queue.pop())

    if kind == "lifo":
        popped.reverse()
    assert pushed == popped


@pytest.mark.parametrize(
    "queue_type,values,expected",
    [
        (
            _types.PriorityQueueMaxHeap,
            [(1, 2, 1,), (7, 8, 5,), (9, 3, 3,)],
            [(7, 8, 5,), (9, 3, 3,), (1, 2, 1,)],
        ),
        (
            _types.PriorityQueueExtMaxHeap,
            [(1, 2, 1,), (7, 8, 5,), (9, 3, 3,)],
            [(7, 8, 5,), (9, 3, 3,), (1, 2, 1,)],
        ),
    ]
)
def test_use_prio_queue(queue_type, values, expected):
    queue = queue_type()
    assert queue.is_empty()

    for a, b, w in values:
        queue.push(a, b, w)

    assert not queue.is_empty()
    assert queue.size() == len(values)

    for ea, eb, ew in expected:
        a, b, w = queue.pop()
        assert (ea, eb, ew) == (a, b, w)

    assert queue.is_empty()

    for a, b, w in values:
        queue.push(a, b, w)

    queue.reset()
    assert queue.is_empty()
