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
    from . import plot
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
from commonnn import _bundle
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
            **recipe_kwargs: Passed on to override entries in the base `recipe`. Use double
                underscores in key names instead of dots, e.g. fitter__na instead of fitter.na.
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
        return self._bundle._input_data.to_components_array()

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
            f"input_data={self._bundle._input_data if self._bundle is not None else None}",
            f"fitter={self._fitter}",
            f"hierarchical_fitter={self._hierarchical_fitter}",
            f"predictor={self._predictor}",
        ])

        return f"{type(self).__name__}({attr_str})"

    def __getitem__(self, key):
        return self._bundle.get_child(key)

    def __setitem__(self, key, value):
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
            if bundle._summary is None:
                bundle._summary = Summary(record_type=self._fitter._record_type)

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


    def trim(self, Bundle bundle = None, protocol="shrinking", **kwargs):

        if bundle is None:
            bundle = self._bundle

        if callable(protocol):
            protocol(bundle, **kwargs)
            return

        if protocol == "shrinking":
            _bundle.trim_shrinking(bundle)
        elif protocol == "trivial":
            _bundle.trim_trivial(bundle)
        elif protocol == "lonechild":
            _bundle.trim_lonechild(bundle)
        elif protocol == "small":
            _bundle.trim_small(bundle, **kwargs)
        else:
            raise ValueError(f"Unknown protocol {protocol}")


    def isolate(
            self,
            bint purge: bool = True,
            bint isolate_input_data: bool = True,
            bundle=None) -> None:
        """Create child clusterings from cluster labels

        Args:
            purge: If `True`, creates a new mapping for the children of this
                clustering.
            isolate_input_data: If `True`, attaches a subset of the input data
                of this clustering to the child.
            bundle: A bundle to operate on. If `None` uses the root bundle.
        """

        if bundle is None:
            bundle = self._bundle

        bundle.isolate(purge, isolate_input_data)

        return

    def reel(
            self,
            depth: Optional[int] = None,
            bundle=None) -> None:

        if bundle is None:
            bundle = self._bundle

        if depth is None:
            depth = UINT_MAX

        bundle.reel(depth)


    def summarize(
            self,
            ax=None,
            quantity: str = "execution_time",
            treat_nan: Optional[Any] = None,
            convert: Optional[Any] = None,
            ax_props: Optional[dict] = None,
            contour_props: Optional[dict] = None,
            plot_style: str = "contourf",
            bundle=None):
        """Generate a 2D plot of record values

        Record values ("time", "clusters", "largest", "noise") are
        plotted against cluster parameters (radius cutoff *r*
        and cnn cutoff *c*).

        Args:
            ax: Matplotlib Axes to plot on.  If `None`, a new Figure
                with Axes will be created.
            quantity: Record value to visualise:
                * "time"
                * "clusters"
                * "largest"
                * "noise"
            treat_nan: If not `None`, use this value to pad nan-values.
            ax_props: Used to style `ax`.
            contour_props: Passed on to contour.
        """

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if bundle is None:
            bundle = self._bundle

        if (self._bundle._summary is None) or (len(self._bundle._summary._list) == 0):
            raise LookupError(
                "No records in summary"
                )

        ax_props_defaults = {
            "xlabel": "$r$",
            "ylabel": "$c$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        contour_props_defaults = {
                "cmap": mpl.cm.inferno,
            }

        if contour_props is not None:
            contour_props_defaults.update(contour_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plotted = plot.plot_summary(
            ax, bundle._summary.to_DataFrame(),
            quantity=quantity,
            treat_nan=treat_nan,
            convert=convert,
            contour_props=contour_props_defaults,
            plot_style=plot_style,
            )

        ax.set(**ax_props_defaults)

        return plotted

    def pie(self, ax=None, pie_props=None, bundle=None):
        """Make a pie plot of the cluster hierarchy based on assigned labels"""

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if bundle is None:
            bundle = self._bundle

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plot.pie(bundle, ax=ax, pie_props=pie_props)

        return

    def tree(
            self,
            ax=None,
            ignore=None,
            pos_props=None,
            draw_props=None,
            bundle=None):
        """Make a layer plot of the cluster hierarchy"""

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if bundle is None:
            bundle = self._bundle

        graph = self.to_nx_DiGraph(ignore=ignore, bundle=bundle)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        pos_props_defaults = {
            "source": "1",
        }

        if pos_props is not None:
            pos_props_defaults.update(pos_props)

        shortened_labels = {}
        for key in graph.nodes.keys():
            skey = key.rsplit(".", 1)
            shortened_labels[key] = skey[len(skey) - 1]

        draw_props_defaults = {
            "labels": shortened_labels,
            "with_labels": True,
            "node_shape": "s",
            "edgecolors": "k",
        }

        if draw_props is not None:
            draw_props_defaults.update(draw_props)

        plot.plot_graph_sugiyama_straight(
            graph, ax=ax,
            pos_props=pos_props_defaults,
            draw_props=draw_props_defaults,
        )

        return

    def evaluate(
            self,
            ax=None,
            clusters: Optional[Container[int]] = None,
            original: bool = False,
            plot_style: str = 'dots',
            parts: Optional[Tuple[Optional[int]]] = None,
            points: Optional[Tuple[Optional[int]]] = None,
            dim: Optional[Tuple[int, int]] = None,
            mask: Optional[Sequence[Union[bool, int]]] = None,
            ax_props: Optional[dict] = None,
            annotate: bool = True,
            annotate_pos: Union[str, dict] = "mean",
            annotate_props: Optional[dict] = None,
            plot_props: Optional[dict] = None,
            plot_noise_props: Optional[dict] = None,
            hist_props: Optional[dict] = None,
            free_energy: bool = True,
            bundle=None):

        """Make 2D plot of an original data set or a cluster result

        Args:
            ax: The `Axes` instance to which to add the plot.  If
            `None`, a new `Figure` with `Axes` will be created.
            clusters:
                Cluster numbers to include in the plot.  If `None`,
                consider all.
            original:
                Allows to plot the original data instead of a cluster
                result.  Overrides `clusters`.  Will be considered
                `True`, if no cluster result is present.
            plot_style:
                The kind of plotting method to use:
                    * "dots", :func:`ax.plot`
                    * "scatter", :func:`ax.scatter`
                    * "contour", :func:`ax.contour`
                    * "contourf", :func:`ax.contourf`
            parts:
                Use a slice (start, stop, stride) on the data parts
                before plotting. Will be applied before a slice on `points`.
            points:
                Use a slice (start, stop, stride) on the data points
                before plotting.
            dim:
                Use these two dimensions for plotting.  If `None`, uses
                (0, 1).
            mask:
                Sequence of boolean or integer values used for optional
                fancy indexing on the point data array.  Note, that this
                is applied after regular slicing (e.g. via `points`) and
                requires a copy of the indexed data (may be slow and
                memory intensive for big data sets).
            annotate:
                If there is a cluster result, plot the cluster numbers.
                Uses `annotate_pos` to determinte the position of the
                annotations.
            annotate_pos:
                Where to put the cluster number annotation. Can be one of:
                    * "mean", Use the cluster mean
                    * "random", Use a random point of the cluster
                    * dict `{1: (x, y), ...}`, Use a specific coordinate
                        tuple for each cluster. Omitted labels will be placed
                        randomly.

            annotate_props:
                Dictionary of keyword arguments passed to
                :func:`ax.annotate`.
            ax_props:
                Dictionary of `ax` properties to apply after
                plotting via :func:`ax.set(**ax_props)`.  If `None`,
                uses defaults that can be also defined in
                the configuration file (*Note yet implemented*).
            plot_props:
                Dictionary of keyword arguments passed to various
                functions (:func:`plot.plot_dots` etc.) with different
                meaning to format cluster plotting.  If `None`, uses
                defaults that can be also defined in
                the configuration file (*Note yet implemented*).
            plot_noise_props:
                Like `plot_props` but for formatting noise point
                plotting.
            hist_props:
               Dictionary of keyword arguments passed to functions that
               involve the computing of a histogram via
               `numpy.histogram2d`.
            free_energy:
                If `True`, converts computed histograms to pseudo free
                energy surfaces.

        Returns:
            Figure, Axes and a list of plotted elements

        Note:
            Requires coordinate access on the input data via
            :meth:`~_types.InputData.to_components_array`.
            Also requires :meth:`~_types.InputData.by_parts` if
            option `parts` is used.
        """

        if bundle is None:
            bundle = self._bundle

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if (bundle._input_data is None) or (
                not bundle._input_data.meta.get("access_components", False)):
            raise ValueError(
                "No data point coordinates found to evaluate."
            )

        if dim is None:
            dim = (0, 1)
        elif dim[1] < dim[0]:
            dim = dim[::-1]  # Problem with wraparound=False?

        if parts is not None:
            by_parts = list(bundle._input_data.by_parts())[slice(*parts)]
            data = np.vstack(by_parts)
        else:
            data = bundle._input_data.to_components_array()

        if points is None:
            points = (None, None, None)

        # Slicing without copying
        data = data[
            slice(*points),
            slice(dim[0], dim[1] + 1, dim[1] - dim[0])
            ]

        if mask is not None:
            data = data[mask]

        # Plot original set or points per cluster?
        cluster_map = None
        if not original:
            if bundle._labels is not None:
                cluster_map = bundle._labels.mapping
                if clusters is None:
                    clusters = list(cluster_map.keys())
            else:
                original = True

        ax_props_defaults = {
            "xlabel": "$x$",
            "ylabel": "$y$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        annotate_props_defaults = {}

        if annotate_props is not None:
            annotate_props_defaults.update(annotate_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if plot_style == "dots":
            plot_props_defaults = {
                'lw': 0,
                'marker': '.',
                'markersize': 5,
                'markeredgecolor': 'none',
                }

            if plot_props is not None:
                plot_props_defaults.update(plot_props)

            plot_noise_props_defaults = {
                'color': 'none',
                'lw': 0,
                'marker': '.',
                'markersize': 4,
                'markerfacecolor': 'k',
                'markeredgecolor': 'none',
                'alpha': 0.3
                }

            if plot_noise_props is not None:
                plot_noise_props_defaults.update(plot_noise_props)

            plotted = plot.plot_dots(
                ax=ax, data=data, original=original,
                cluster_map=cluster_map,
                clusters=clusters,
                dot_props=plot_props_defaults,
                dot_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        elif plot_style == "scatter":
            plot_props_defaults = {
                's': 10,
            }

            if plot_props is not None:
                plot_props_defaults.update(plot_props)

            plot_noise_props_defaults = {
                'color': 'k',
                's': 10,
                'alpha': 0.5
            }

            if plot_noise_props is not None:
                plot_noise_props_defaults.update(plot_noise_props)

            plotted = plot.plot_scatter(
                ax=ax, data=data, original=original,
                cluster_map=cluster_map,
                clusters=clusters,
                scatter_props=plot_props_defaults,
                scatter_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        if plot_style in ["contour", "contourf", "histogram"]:

            hist_props_defaults = {
                "avoid_zero_count": False,
                "mass": True,
                "mids": True
            }

            if hist_props is not None:
                hist_props_defaults.update(hist_props)

            if plot_style == "contour":

                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_contour(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot_style == "contourf":
                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_contourf(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot_style == "histogram":
                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_histogram2d(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

        ax.set(**ax_props_defaults)

        return plotted

    def to_nx_DiGraph(self, ignore=None, bundle=None):
        """Convert cluster hierarchy to networkx DiGraph

        Keyword args:
            ignore: A set of label not to include into the graph.  Use
                for example to exclude noise (label 0).
            bundle: The bundle to start with. If `None`, uses the root bundle.
        """

        if not NX_FOUND:
            raise ModuleNotFoundError("No module named 'networkx'")

        if bundle is None:
            bundle = self._bundle

        def add_children(clustering_label, clustering, graph):
            for child_label, child_clustering in sorted(clustering._children.items()):

                if child_label in ignore:
                    continue

                padded_child_label = ".".join([clustering_label, str(child_label)])
                graph.add_node(padded_child_label, object=child_clustering)
                graph.add_edge(clustering_label, padded_child_label)

                if child_clustering._children:
                    add_children(padded_child_label, child_clustering, graph)

        if ignore is None:
            ignore = set()

        if not isinstance(ignore, set):
            ignore = set(ignore)

        graph = nx.DiGraph()
        graph.add_node("1", object=bundle)
        add_children("1", bundle, graph)

        return graph

    def to_dtrajs(self, bundle=None):
        """Convert cluster label assignments to discrete state  trajectory"""

        if bundle is None:
            bundle = self._bundle

        labels_array = bundle.labels
        if labels_array is None:
            return []

        edges = None
        if bundle._input_data is not None:
            edges = self._input_data.meta.get("edges")

        if edges is None:
            return [labels_array]

        dtrajs = np.split(labels_array, np.cumsum(edges))

        last_dtraj_index = len(dtrajs) - 1
        if len(dtrajs[last_dtraj_index]) == 0:
            dtrajs = dtrajs[:last_dtraj_index]

        return dtrajs
