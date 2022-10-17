from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from collections import Counter
import functools
import time

try:
    import pandas as pd
    PANDAS_FOUND = True
    PD_DTYPE_MAP = {int: pd.Int64Dtype(), float: pd.Float64Dtype()}
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    PANDAS_FOUND = False


class Record(ABC):
    """Cluster result container

    :obj:`~commonnn.cluster.Record` instances can be created during
    :meth:`~commonnn.cluster.Clustering.fit` and
    are collected in :obj:`~commonnn.cluster.Summary`.
    """

    __slots__ = []
    _dtypes = []

    def __init__(self, *args):

        for slot, arg in zip(self.__slots__, args):
            setattr(self, slot, arg)

    def __repr__(self):
        attrs_str = ", ".join([
            f"{attr}={getattr(self, attr)!r}"
            for attr in self.__slots__
        ])
        return f"{type(self).__name__}({attrs_str})"

    def to_dict(self):
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__
        }

    @classmethod
    @abstractmethod
    def from_bundle(cls, bundle, cluster_params=None, **kwargs):
        """Create record from (clustered) bundle data"""


class CommonNNRecord(Record):
    """Cluster result container

    :obj:`~commonnn.cluster.Record` instances can be created during
    :meth:`~commonnn.cluster.Clustering.fit` and
    are collected in :obj:`~commonnn.cluster.Summary`.
    """

    __slots__ = [
        "n_points",
        "radius_cutoff",
        "similarity_cutoff",
        "member_cutoff",
        "max_clusters",
        "n_clusters",
        "ratio_largest",
        "ratio_noise",
        "execution_time",
    ]
    _dtypes = [
        int,    # points
        float,  # r
        int,    # n
        int,    # min
        int,    # max
        int,    # clusters
        float,  # largest
        float,  # noise
        float,  # time
    ]

    def __str__(self):
        attr_str = ""
        for attr in self.__slots__:
            if attr == "execution_time":
                continue

            value = getattr(self, attr)
            if value is None:
                attr_str += f"{value!r:<10}"
            elif isinstance(value, float):
                attr_str += f"{value:<10.3f}"
            else:
                attr_str += f"{value:<10}"

        if self.execution_time is not None:
            hours, rest = divmod(self.execution_time, 3600)
            minutes, seconds = divmod(rest, 60)
            execution_time_str = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:.3f}"
        else:
            execution_time_str = None

        printable = (
            f'{"-" * 95}\n'
            f"#points   "
            f"r         "
            f"nc        "
            f"min       "
            f"max       "
            f"#clusters "
            f"%largest  "
            f"%noise    "
            f"time     \n"
            f"{attr_str}"
            f"{execution_time_str}\n"
            f'{"-" * 95}\n'
        )
        return printable

    @classmethod
    def from_bundle(cls, bundle, cluster_params=None, **kwargs):
        n_noise = 0
        frequencies = Counter(bundle._labels.labels)

        if 0 in frequencies:
            n_noise = frequencies.pop(0)

        n_largest = frequencies.most_common(1)[0][1] if frequencies else 0

        if cluster_params is not None:
            params = cluster_params.to_dict()
        else:
            params = {}

        params.update(kwargs)

        return cls(
            bundle._input_data.n_points,
            params.get("radius_cutoff"),
            params.get("similarity_cutoff"),
            params.get("member_cutoff"),
            params.get("max_clusters"),
            len(bundle._labels.to_set() - {0}),
            n_largest / bundle._input_data.n_points,
            n_noise / bundle._input_data.n_points,
            params.get("execution_time"),
        )


class Summary(MutableSequence):
    """List like container for cluster results

    Stores instances of :obj:`~commonnn.report.Record`.
    """

    def __init__(self, iterable=None, *, record_type=Record):
        if iterable is None:
            iterable = []

        self._record_type = record_type

        self._list = []
        for i in iterable:
            self.append(i)

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, item):
        if type(item) == self._record_type:
            self._list.__setitem__(key, item)
        else:
            raise TypeError(
                f"Summary can only contain records of type {self._record_type.__name__}"
            )

    def __delitem__(self, key):
        self._list.__delitem__(key)

    def __len__(self):
        return self._list.__len__()

    def __str__(self):
        return self._list.__str__()

    def insert(self, index, item):
        if type(item) == self._record_type:
            self._list.insert(index, item)
        else:
            raise TypeError(
                f"Summary can only contain records of type {self._record_type.__name__}"
            )

    def to_DataFrame(self, map_pd_dtypes=True):
        """Convert list of records to (typed) :obj:`pandas.DataFrame`

        Returns:
            :obj:`pandas.DataFrame`
        """

        if not PANDAS_FOUND:
            raise ModuleNotFoundError("No module named 'pandas'")

        _record_dtypes = self._record_type._dtypes

        if map_pd_dtypes:
            _record_dtypes = [PD_DTYPE_MAP.get(x, x) for x in _record_dtypes]

        content = []
        for field in self._record_type.__slots__:
            content.append([
                record.__getattribute__(field)
                for record in self._list
            ])

        return make_typed_DataFrame(
            columns=self._record_type.__slots__,
            dtypes=_record_dtypes,
            content=content,
        )


def make_typed_DataFrame(columns, dtypes, content=None):
    """Construct :obj:`pandas.DataFrame` with typed columns"""

    if not PANDAS_FOUND:
        raise ModuleNotFoundError("No module named 'pandas'")

    assert len(columns) == len(dtypes)

    if content is None:
        content = [[] for i in range(len(columns))]

    df = pd.DataFrame({
        k: pd.array(c, dtype=v)
        for k, v, c in zip(columns, dtypes, content)
    })

    return df


def timed(function):
    """Decorator to measure execution time"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        go = time.time()
        wrapped_return = function(*args, **kwargs)
        stop = time.time()

        return wrapped_return, stop - go
    return wrapper
