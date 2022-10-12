from collections.abc import MutableMapping
import functools
from operator import itemgetter
import time
from typing import Any, Optional, Type, Union
from typing import Container, Iterable, List, Tuple, Sequence
import weakref

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # from . import plot
    MPL_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    MPL_FOUND = False

try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    NX_FOUND = False

import numpy as np
cimport numpy as np

try:
    import pandas as pd
    PANDAS_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    PANDAS_FOUND = False

from commonnn._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from commonnn._fit import Fitter, HierarchicalFitter, Predictor
from commonnn._types import InputData
from commonnn import recipes
from commonnn.report import Record, Summary


class Clustering:
    r"""Organises a clustering

    Aggregates all necessary types to
    carry out a clustering of input data points.
    """

    def __init__(
            self,
            data=None, *,  # TODO use positional only modifier "/" (Python >= 3.8)
            fitter=None,
            hierarchical_fitter=None,
            predictor=None,
            bundle_kwargs=None,
            recipe=None,
            **recipe_kwargs):
        """
        Keyword args:
            data:
                The data points to be clustered. Can be one of
                    * `None`:
                        Plain initialisation without input data.
                    * A :class:`~commonnn._bundle.Bundle`:
                        Initialisation with a ready-made input data bundle.
                    * Any object implementing the input data interface
                    (see :class:`~commonnn._types.InputData` or
                    :class:`~commonnn._types.InputDataExtInterface`):
                        in this case, additional keyword arguments can be passed
                        via `bundle_kwargs` which are used to initialise a
                        :class:`~commonnn._bundle.Bundle` from the input data,
                        e.g. `labels`, `children`, etc.
                    * Raw input data: Takes the input data type and a preparation
                    hook from the `recipe` and wraps the raw data.
            fitter:
                Executes the clustering procedure. Can be
                    * Any object implementing the fitter interface (see :class:`~commonnn._fit.Fitter` or
                    :class:`~commonnn._fit.FitterExtInterface`).
                    * None:
                        In this case, the fitter is tried to be build from the `recipe` or left
                        as `None`.
            hierarchical_fitter:
                Like `fitter` but for hierarchical clustering (see
                :class:`~commonnn._fit.HierarchicalFitter` or
                :class:`~commonnn._fit.HierarchicalFitterExtInterface`).
            predictor:
                Translates a clustering result from one bundle to another. Treated like
                `fitter` (see
                :class:`~commonnn._fit.Predictor` or
                :class:`~commonnn._fit.PredictorExtInterface`).
            bundle_kwargs: Used to create a :class:`~commonnn._bundle.Bundle`
                if `data` is neither a bundle nor `None`.
            recipe:
                Used to assemble a fitter etc. and to wrap raw input data. Can be
                    * A string corresponding to a registered default recipe (see
                        :obj:`~commonnn.recipes.REGISTERED_RECIPES`
                    )
                    * A recipe, i.e. a mapping of component keywords to component types
        """

        builder = recipes.Builder(recipe, **recipe_kwargs)

        if bundle_kwargs is None:
            bundle_kwargs = {}

        if data is None:
            self._bundle = None

        elif isinstance(data, Bundle):
            self._bundle = data

        elif isinstance(data, InputData):
            if bundle_kwargs is None:
                bundle_kwargs = {}

            self._bundle = Bundle(data, **bundle_kwargs)
        else:
            # TODO: Guess input data type and preparation hook
            data = builder.make_input_data(data)
            self._bundle = Bundle(data, **bundle_kwargs)

        for kw, component_kw, kw_type in [
                (fitter, "fitter", Fitter),
                (hierarchical_fitter, "hierarchical_fitter", HierarchicalFitter),
                (predictor, "predictor", Predictor)
                ]:

            if isinstance(kw, kw_type):
                setattr(self, f"_{component_kw}", kw)
                continue

            if kw is not None:
                builder.recipe[component_kw] = kw

            kw = builder.make_component(component_kw)
            if kw is object: kw = None
            setattr(self, f"_{component_kw}", kw)

    @property
    def root(self):
        if self._bundle is None:
            return
        return self._bundle

    @property
    def labels(self):
        """
        Direct access to :obj:`~commonnn._types.Labels.labels`
        holding cluster label assignments for points in :obj:`~commonnn._types.InputData`,
        stored on the root :obj:`~commonnn._bundle.Bundle`.
        """
        if self._bundle is None:
            return None
        if self._bundle._labels is None:
            return None
        return self._bundle._labels.labels

    @labels.setter
    def labels(self, value):
        """
        Direct access to :obj:`~commonnn._types.Labels`
        holding cluster label assignments for points in :obj:`~commonnn._types.InputData`,
        stored on the root :obj:`~commonnn._bundle.Bundle`.
        """
        if self._bundle is None:
            raise ValueError("Can't set labels because there is no root bundle")
        self._bundle.labels = value

    @property
    def input_data(self):
        if self._bundle is None:
            return None
        if self._bundle._input_data is None:
            return None
        return self._bundle._input_data.data

    @property
    def fitter(self):
        return self._fitter

    @fitter.setter
    def fitter(self, value):
        if (value is not None) & (not isinstance(value, Fitter)):
            raise TypeError(
                f"Can't use object of type {type(value).__name__} as fitter. "
                f"Expected type {Fitter.__name__}."
                )
        self._fitter = value

    @property
    def children(self):
        """
        Return a mapping of child cluster labels to
        :obj:`commonnn._bundle.Bundle` instances representing
        the children of this clustering (the root bundle).
        """
        return self._bundle._children

    @property
    def summary(self):
        """
        Return an instance of :obj:`commonnn.report.Summary`
        collecting clustering result records for this clustering
        (the root bundle).
        """
        return self._bundle._summary

    def __str__(self):
        attr_str = ", ".join([
            f"input_data={self._bundle._input_data}",
            f"fitter={self._fitter}",
            f"hierarchical_fitterh={self._hierarchical_fitter}",
            f"predictor={self._predictor}",
        ])

        return f"{type(self).__name__}({attr_str})"

    def __getitem__(self, key):
        return self._bundle.get_child(key)

    def __setitem___(self, key, value):
        self._bundle.add_child(key, value)

    def _fit(
            self,
            cluster_params: Type["ClusterParameters"],
            bundle=None) -> None:
        """Execute clustering procedure

        Low-level alternative to
        :meth:`~comonnn.cluster.Clustering.fit`.

        Note:
            No pre-processing of cluster parameters (radius and
            similarity value adjustments based on used metric,
            neighbours getter, etc.) and label initialisation is done
            before the fit. It is the users responsibility to take care
            of this to obtain sensible results.
        """

        if bundle is None:
            bundle = self._bundle

        self._fitter._fit(bundle._input_data, bundle._labels, cluster_params)

    def fit(
            self, bundle=None, *,
            sort_by_size=True,
            member_cutoff=None,
            max_clusters=None,
            record=True,
            v=True,
            **kwargs) -> None:
        """Execute clustering procedure

        Keyword args:
            sort_by_size: Weather to sort (and trim) the created
                :obj:`Labels` instance.  See also
                :meth:`Labels.sort_by_size`.
            member_cutoff: Valid clusters need to have at least this
                many members.  Passed on to :meth:`Labels.sort_by_size`
                if `sort_by_size` is `True`.  Has no effect otherwise
                and valid clusters have at least one member.
            max_clusters: Keep only the largest `max_clusters` clusters.
                Passed on to :meth:`Labels.sort_by_size` if
                `sort_by_size` is `True`.  Has no effect otherwise.
            record: Whether to create a :obj:`Record`
                instance for this clustering which is appended to the
                :obj:`Summary` of the clustered bundle.
            v: Be chatty.

        Note:
            Further keyword arguments are passed on to
            :obj:`~commonnn._fit.Fitter.fit`.
        """

        if bundle is None:
            bundle = self._bundle

        execution_time, cluster_params = self._fitter.fit(bundle, **kwargs)

        if sort_by_size:
            bundle._labels.sort_by_size(member_cutoff, max_clusters)

        if record:
            rec = self._fitter._record_type.from_bundle(
                bundle, cluster_params,
                member_cutoff=member_cutoff,
                max_clusters=max_clusters,
                execution_time=execution_time
                )

            if v:
                print(rec)

            bundle._summary.append(rec)

    def fit_hierarchical(
            self,
            *args,
            purge=True,
            bundle=None,
            **kwargs):
        """Execute hierarchical clustering procedure

        Keyword args:
            purge: Reset children dictionary of root bundle
            bundle: Root bundle

        Note:
            Used arguments and further keyword arguments depend on the
            used hierarchical fitter.
        """

        if bundle is None:
            bundle = self._bundle

        if purge or (bundle._children is None):
            bundle._children = {}

        self._hierarchical_fitter.fit(bundle, *args, **kwargs)

    def predict(
            self, other, *, bundle=None, **kwargs):
        """Execute prediction procedure

        Args:
            other: :obj:`commonnn._bundle.Bundle` instance for
                which cluster labels should be predicted.

        Keyword args:
            bundle: Bundle to predict from. If None, uses the root bundle.
        """

        if bundle is None:
            bundle = self._bundle

        self._predictor.predict(bundle, other, **kwargs)
