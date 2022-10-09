import time

try:
    import pandas as pd
    PANDAS_FOUND = True
except ModuleNotFoundError:
    PANDAS_FOUND = False

import pytest

from commonnn import report


pytestmark = pytest.mark.pandas


def test_make_typed_DataFrame():
    if not PANDAS_FOUND:
        pytest.skip("Test function requires pandas")

    tdf = report.make_typed_DataFrame(
        columns=["a", "b"],
        dtypes=[int, str],
        content=[[0, 1, 2], ["None", "True", "foo"]],
    )
    assert list(tdf.columns) == ["a", "b"]
    assert len(tdf) == 3
    assert tdf["a"].dtype == int
    assert tdf["b"].dtype == 'O'

    tdf = report.make_typed_DataFrame(
        columns=["a", "b"],
        dtypes=[int, str],
        content=[[0, 1, 2], ["None", "True", "foo"]],
    )
    assert list(tdf.columns) == ["a", "b"]
    assert len(tdf) == 3
    assert tdf["a"].dtype == int
    assert tdf["b"].dtype == 'O'


def test_make_empty_typed_DataFrame():
    if not PANDAS_FOUND:
        pytest.skip("Test function requires pandas")

    tdf = report.make_typed_DataFrame(
        columns=["a", "b"],
        dtypes=[int, str],
    )
    assert list(tdf.columns) == ["a", "b"]
    assert len(tdf) == 0


def test_record_init(file_regression):
    record = report.CommonNNRecord(10, 0.1, 2, 1, None, 3, 0.5, 0.05, None)
    without_extime = str(record)
    record = report.CommonNNRecord(10, 0.1, 2, 1, None, 3, 0.5, 0.05, 0.0001)
    file_regression.check(
        f"{record!r}\n\n{record.to_dict()}\n\n{record!s}\n{without_extime}"
    )


def test_summary_init():
    summary = report.Summary()
    assert len(summary) == 0
    assert str(summary) == '[]'

    summary = report.Summary(
        [report.CommonNNRecord() for _ in range(5)],
        record_type=report.CommonNNRecord
    )
    assert len(summary) == 5

    with pytest.raises(TypeError):
        summary[0] = report.Record()
    summary[0] = report.CommonNNRecord()

    assert isinstance(summary[0], report.CommonNNRecord)

    del summary[0]
    assert len(summary) == 4

    with pytest.raises(TypeError):
        summary.insert(2, report.Record())
    summary.insert(2, report.CommonNNRecord())
    assert len(summary) == 5


def test_summary_to_DataFrame():
    if not PANDAS_FOUND:
        pytest.skip("Test function requires pandas")

    summary = report.Summary(
        [report.CommonNNRecord(*[1] * 9) for _ in range(5)],
        record_type=report.CommonNNRecord
    )

    tdf = summary.to_DataFrame()
    assert len(tdf) == 5
    assert tdf["n_points"][0] == 1


def test_timed_decorator():
    def some_function():
        time.sleep(0.01)

    decorated = report.timed(some_function)
    decorated_result = decorated()

    assert isinstance(decorated_result, tuple)
    assert decorated_result[0] is None
    assert isinstance(decorated_result[1], float)
