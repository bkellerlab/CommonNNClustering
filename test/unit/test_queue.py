import pytest

from commonnn import _types


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
    ]
)
def test_use_prio_queue(queue_type, values, expected):
    queue = queue_type()

    assert queue.is_empty()

    for a, b, w in values:
        queue.push(a, b, w)

    for ea, eb, ew in expected:
        a, b, w = queue.pop()
        assert (ea, eb, ew) == (a, b, w)

    assert queue.is_empty()