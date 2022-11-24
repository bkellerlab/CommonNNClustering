"""User module containing utilities for plotting"""

from collections import deque
import random
from typing import Dict, Tuple
from typing import Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt

try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    NX_FOUND = False

import numpy as np

try:
    import scipy.signal
    import scipy.interpolate
    SCIPY_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    SCIPY_FOUND = False


def traverse_graph_dfs_children_first(graph, source):
    """Yield nodes from networkx graph beginning with deeper nodes

    Example:
        >>> import networkx
        >>> g = networkx.DiGraph({0: [1, 2], 1: [3, 4], 2: [5, 6]})
        >>> # Child nodes retrieved before their parents
        >>> list(traverse_graph_dfs_like(g, 0))
        [3, 4, 1, 5, 6, 2, 0]
        >>> # As opposed to classic depth-first-search
        >>> list(networkx.algorithms.dfs_tree(g, 0))
        [0, 1, 3, 4, 2, 5, 6]

    Args:
        graph: A networkx graph.
        source: The source node in `graph` from which to start the
            traversal.

    Yields:
        Nodes

    Note:
        Equivalent traversal is also implemented as
        `networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes`_.

    .. _networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes:
        https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes.html
    """

    stack = []
    stack.append(source)
    child_generators = {source: graph.neighbors(source)}

    while stack:
        try:
            child = next(child_generators[stack[-1]])
        except StopIteration:
            yield stack.pop()
        else:
            stack.append(child)
            child_generators[child] = graph.neighbors(child)


def find_node_positions_sugiyama_straight(
        graph, source=None, x_spacing=0.1, y_spacing=0.1):
    """Find suitable positions for plotting the nodes of a graph in 2D

    Yields a layered graph (Sugiyama- or dot-like style) in which the
    leaf nodes are separated in x-direction by `x_spacing`. Parent nodes
    will be placed symmetrically above their child nodes.

    Args:
        graph: A networkx graph. The layout only makes sense for directed
            graphs representing a hierarchical (rooted) tree (this is
            not checked). Nodes are
            assumed to be strings of integers (cluster labels) separated
            by dots, e.g. "1" (tree root), "1.1", "1.2" (1st gen. childs)
            etc.
        source: The source node in `graph` from which to start the
            traversal. If `None`, tries to find the root by picking a
            random node and traversing upwards until a node with no
            incoming edge is found.
        x_spacing: Separation of leaf nodes in x-direction.
        y_spacing: Separation of hierarchy layers in y-direction.

    Returns:
        Dictionary with nodes as keys and positions
        (NumPy array(x, y)) as values.
    """

    if source is None:
        root_candidate = next(iter(graph.nodes))
        visited_nodes = {}

        while graph.in_degree(root_candidate) != 0:
            if root_candidate in visited_nodes:
                raise RuntimeError(
                    "can not find root node of cyclic graph"
                )
            visited_nodes.add(root_candidate)
            root_candidate = next(iter(graph.in_edges(root_candidate)))[0]

        source = root_candidate

    rightmost_x = 0
    previous_level = 0
    positions = {}

    for node in traverse_graph_dfs_children_first(graph, source):
        level = len(node.split(".")) - 1
        if (level >= previous_level) or (graph.out_degree(node) == 0):
            positions[node] = np.array([rightmost_x, -y_spacing * level])
            rightmost_x += x_spacing
        else:
            mean_pos = np.mean([positions[n][0] for n in graph.neighbors(node)])
            positions[node] = np.array([mean_pos, -y_spacing * level])

        previous_level = level

    return positions


def plot_graph_sugiyama_straight(graph, ax, pos_props=None, draw_props=None):

    if not NX_FOUND:
        raise ModuleNotFoundError("No module named 'networkx'")

    if pos_props is None:
        pos_props = {}

    if draw_props is None:
        draw_props = {}

    nx.draw(
        graph,
        pos=find_node_positions_sugiyama_straight(graph, **pos_props),
        ax=ax,
        **draw_props
    )

    return


def get_pieces(bundle):
    """Transform cluster tree to layers of hierarchy levels

    Each hierarchy level will be represented as a list of tuples holding
    a cluster string identifier and the number of points in this cluster.

    Note:
        Used by :meth:`plot_pie`.

    Args:
        bundle: A root instance of :obj:`~commonnn._bundle.Bundle`
    """

    if bundle._labels is None:
        raise LookupError(
            "Root clustering has no labels"
        )

    pieces = [[("1", bundle._labels.n_points)]]
    expected_parent_pool = iter(pieces[-1])
    next_parent_label, next_parent_membercount = next(expected_parent_pool)
    expected_parent_found = False
    pieces.append([])

    terminal_cluster_references = deque([("1", bundle)])
    new_terminal_cluster_references = deque()

    while True:
        parent_label, bundle_instance = terminal_cluster_references.popleft()

        while parent_label != next_parent_label:
            if not expected_parent_found:
                pieces[-1].append((f"{next_parent_label}.0", next_parent_membercount))
            else:
                expected_parent_found = False

            next_parent_label, next_parent_membercount = next(expected_parent_pool)

        expected_parent_found = True

        cluster_shares = [
            (f"{'.'.join([parent_label, str(k)])}", len(v))
            for k, v in sorted(bundle_instance._labels.mapping.items())
        ]

        pieces[-1].extend(cluster_shares)

        if bundle_instance._children:
            for child_label, child_clustering in sorted(
                    bundle_instance._children.items()):

                if child_clustering._labels is None:
                    continue

                new_terminal_cluster_references.append(
                    (
                        f"{'.'.join([parent_label, str(child_label)])}",
                        child_clustering
                    )
                )

        if not terminal_cluster_references:

            for next_parent_label, next_parent_membercount in expected_parent_pool:
                pieces[-1].append((f"{next_parent_label}.0", next_parent_membercount))

            # DEBUG
            assert sum(p[1] for p in pieces[-1]) == bundle._labels.n_points

            if not new_terminal_cluster_references:
                break

            terminal_cluster_references = new_terminal_cluster_references
            new_terminal_cluster_references = deque()
            expected_parent_pool = iter(pieces[-1])
            next_parent_label, next_parent_membercount = next(expected_parent_pool)
            expected_parent_found = False
            pieces.append([])

    return pieces


def pie(clustering, ax, pie_props=None):
    """Illustrate (hierarchical) cluster result as pie diagram

    Args:
        clustering: Root instance of :obj:`commonnn.cluster.Clustering`
            being the origin of the pie diagram.
        ax: Matplotlib `Axes` instance to plot on.
        pie_props: Dictionary passed to :func:`matplotlib.pyplot.pie`.

    Returns:
        List of plotted elements (pie rings)
    """

    if ax is None:
        ax = plt.gca()

    pie_props_defaults = {
        "normalize": False,
        "radius": 0.5,
        "wedgeprops": dict(width=0.5, edgecolor="w"),
    }

    if pie_props is not None:
        pie_props_defaults.update(pie_props)

    radius = pie_props_defaults.pop("radius")
    try:
        _ = pie_props_defaults.pop("colors")
    except KeyError:
        pass

    pieces = get_pieces(clustering)
    n_points = pieces[0][0][1]

    for level, cluster_shares in enumerate(pieces[1:]):
        ax.set_prop_cycle(None)
        ringvalues = []
        colors = []
        for label, member_count in cluster_shares:
            ringvalues.append(member_count / n_points)
            if label.rsplit(".", 1)[-1] == "0":
                colors.append("#262626")
            else:
                colors.append(next(ax._get_lines.prop_cycler)["color"])

        ax.pie(
            ringvalues,
            radius=radius * (level + 1),
            colors=colors,
            **pie_props_defaults,
        )

    return


def plot_summary(
        ax,
        summary,
        quantity="execution_time",
        treat_nan=None,
        convert=None,
        contour_props=None,
        plot_style="contourf"):
    """Generate a 2D plot of record values"""

    if plot_style not in {"contour", "contourf"}:
        raise ValueError(
            'Keyword argument `plot_style` must be one of ["contour", "contourf"]'
        )

    if contour_props is None:
        contour_props = {}

    pivot = summary.groupby(
        ["radius_cutoff", "similarity_cutoff"]
    ).mean()[quantity].reset_index().pivot(
        index="radius_cutoff", columns="similarity_cutoff"
        )

    X_, Y_ = np.meshgrid(pivot.index.values, pivot.columns.levels[1].values)

    values_ = pivot.values.T

    if treat_nan is not None:
        values_[np.isnan(values_)] = treat_nan

    if convert is not None:
        values_ = np.apply_along_axis(convert, 0, values_)

    plotted = []

    if plot_style == "contourf":
        plotted.append(ax.contourf(X_, Y_, values_, **contour_props))
    elif plot_style == "contour":
        plotted.append(ax.contour(X_, Y_, values_, **contour_props))

    return plotted


def plot_histogram(
        ax,
        x,
        maxima: bool = False,
        maxima_props: dict = None,
        annotate_props: dict = None,
        hist_props: dict = None,
        ax_props: dict = None,
        plot_props: dict = None,
        inter_props: dict = None):
    """Plot a histogram from 1D data

    Args:
        ax: Matplotlib Axes to plot on.
        maxima: Whether to mark the maxima of the
            distribution. Uses `scipy.signal.argrelextrema`_.
        maxima_props: Keyword arguments passed to
            `scipy.signal.argrelextrema`_ if `maxima` is set
            to True.
        annotate_props: Keyword arguments passed to
            `ax.annotate` to draw text if `maxima` is set
            to True.
        hist_props: Keyword arguments passed to
            `numpy.histogram`_ to compute the histogram.
        ax_props: Keyword arguments for Matplotlib Axes styling.
        plot_props: Keyword arguments used when plotting histogram line.
        inter_props: Keyword arguments passed on to
        `scipy.interpolate.interp1d`_

    .. _scipy.signal.argrelextrema:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html
    .. _scipy.interpolate.interp1d:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    .. _numpy.histogram:
        https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """

    if not SCIPY_FOUND:
        raise ModuleNotFoundError("No module named 'scipy'")

    hist_props_defaults = {
        "bins": 100,
        "density": True,
    }

    if hist_props is not None:
        hist_props_defaults.update(hist_props)

    histogram, bins = np.histogram(
        x,
        **hist_props_defaults
    )

    binmids = 0.5 * (bins[:-1] + bins[1:])

    if inter_props is not None:
        inter_props_defaults = {
            "ifactor": 0.5,
            "kind": 'linear',
        }

        inter_props_defaults.update(inter_props)

        ifactor = inter_props_defaults.pop("ifactor")

        ipoints = int(
            np.ceil(len(binmids) * ifactor)
        )
        ibinmids = np.linspace(binmids[0], binmids[-1], ipoints)
        histogram = scipy.interpolate.interp1d(
            binmids,
            histogram,
            **inter_props_defaults
        )(ibinmids)

        binmids = ibinmids

    ylimit = np.max(histogram) * 1.1

    ax_props_defaults = {
        "xlabel": "d / au",
        "ylabel": '',
        "yticks": (),
        "xlim": (np.min(binmids), np.max(binmids)),
        "ylim": (0, ylimit),
    }

    if ax_props is not None:
        ax_props_defaults.update(ax_props)

    plot_props_defaults = {}

    if plot_props is not None:
        plot_props_defaults.update(plot_props)

    if ax is None:
        fig, ax = plt.subplots()
    # else:
    #     fig = ax.get_figure()

    line = ax.plot(binmids, histogram, **plot_props_defaults)

    if maxima:
        maxima_props_ = {
            "order": 2,
            "mode": "clip"
        }

        if maxima_props is not None:
            maxima_props_.update(maxima_props)

        annotate_props_ = {}

        if annotate_props is not None:
            annotate_props_.update(annotate_props)

        found = scipy.signal.argrelextrema(
            histogram, np.greater, **maxima_props_
        )[0]

        annotations = []
        for candidate in found:
            annotations.append(
                ax.annotate(
                    f"{binmids[candidate]:.2f}",
                    xy=(binmids[candidate], histogram[candidate]),
                    xytext=(binmids[candidate],
                            histogram[candidate] + (ylimit / 100)),
                    **annotate_props_
                )
            )
    else:
        annotations = None

    ax.set(**ax_props_defaults)

    return line, annotations


def plot_dots(
        ax,
        data,
        original=True,
        cluster_map=None,
        clusters=None,
        dot_props=None,
        dot_noise_props=None,
        annotate=False,
        annotate_pos="mean",
        annotate_props=None):

    if dot_props is None:
        dot_props = {}

    if dot_noise_props is None:
        dot_noise_props = {}

    plotted = []

    if original:
        # Plot the original data
        plotted.append(ax.plot(data[:, 0], data[:, 1], **dot_props))

    if cluster_map is None:
        cluster_map = {}

    else:
        # Loop through the cluster result
        for cluster, cpoints in sorted(cluster_map.items()):
            # plot if cluster is in the list of considered clusters
            if cluster in clusters:
                cpoints = list(cpoints)

                # treat noise differently
                if cluster == 0:
                    plotted.append(
                        ax.plot(
                            data[cpoints, 0], data[cpoints, 1],
                            **dot_noise_props
                        )
                    )

                else:
                    plotted.append(
                        ax.plot(
                            data[cpoints, 0], data[cpoints, 1],
                            **dot_props
                        )
                    )

                    if annotate:
                        plotted.append(
                            annotate_points(
                                ax, annotate_pos, data, cpoints, cluster,
                                annotate_props
                            )
                        )
    return plotted


def plot_scatter(
        ax,
        data,
        original=True,
        cluster_map=None,
        clusters=None,
        scatter_props=None,
        scatter_noise_props=None,
        annotate=False,
        annotate_pos="mean",
        annotate_props=None):

    if scatter_props is None:
        scatter_props = {}

    if scatter_noise_props is None:
        scatter_noise_props = {}

    plotted = []

    if original:
        plotted.append(ax.scatter(data[:, 0], data[:, 1], **scatter_props))

    if cluster_map is None:
        cluster_map = {}

    else:
        for cluster, cpoints in sorted(cluster_map.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                # treat noise differently
                if cluster == 0:
                    plotted.append(
                        ax.scatter(
                            data[cpoints, 0], data[cpoints, 1],
                            **scatter_noise_props
                        )
                    )

                else:
                    plotted.append(
                        ax.scatter(
                            data[cpoints, 0], data[cpoints, 1],
                            **scatter_props
                        )
                    )

                    if annotate:
                        plotted.append(
                            annotate_points(
                                ax, annotate_pos, data, cpoints, cluster,
                                annotate_props
                            )
                        )
    return plotted


def plot_contour(
        ax,
        data,
        original=True,
        cluster_map=None,
        clusters=None,
        contour_props=None,
        contour_noise_props=None,
        hist_props=None,
        free_energy=True,
        annotate=False,
        annotate_pos="mean",
        annotate_props=None):

    if contour_props is None:
        contour_props = {}

    if contour_noise_props is None:
        contour_noise_props = {}

    if hist_props is None:
        hist_props = {}

    if "avoid_zero_count" in hist_props:
        avoid_zero_count = hist_props["avoid_zero_count"]
        del hist_props["avoid_zero_count"]
    else:
        avoid_zero_count = False

    if "mass" in hist_props:
        mass = hist_props["mass"]
        del hist_props["mass"]
    else:
        mass = True

    if "mids" in hist_props:
        mids = hist_props["mids"]
        del hist_props["mids"]
    else:
        mids = True

    plotted = []

    if original:
        x_, y_, H = get_histogram2d(
            data[:, 0],
            data[:, 1],
            mids=mids,
            mass=mass,
            avoid_zero_count=avoid_zero_count,
            hist_props=hist_props,
        )

        if free_energy:
            H = get_free_energy(H)

        X, Y = np.meshgrid(x_, y_)
        plotted.append(ax.contour(X, Y, H, **contour_props))
    else:
        if cluster_map is None:
            cluster_map = {}

        for cluster, cpoints in sorted(cluster_map.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                x_, y_, H = get_histogram2d(
                    data[cpoints, 0],
                    data[cpoints, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props,
                )

                if free_energy:
                    H = get_free_energy(H)

                if cluster == 0:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(ax.contour(X, Y, H, **contour_noise_props))
                else:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(ax.contour(X, Y, H, **contour_props))

                if annotate:
                    plotted.append(
                        annotate_points(
                            ax, annotate_pos, data, cpoints, cluster,
                            annotate_props
                        )
                    )

    return plotted


def plot_contourf(
        ax,
        data,
        original=True,
        cluster_map=None,
        clusters=None,
        contour_props=None,
        contour_noise_props=None,
        hist_props=None,
        free_energy=True,
        annotate=False,
        annotate_pos="mean",
        annotate_props=None):

    if contour_props is None:
        contour_props = {}

    if contour_noise_props is None:
        contour_noise_props = {}

    if hist_props is None:
        hist_props = {}

    if "avoid_zero_count" in hist_props:
        avoid_zero_count = hist_props["avoid_zero_count"]
        del hist_props["avoid_zero_count"]
    else:
        avoid_zero_count = False

    if "mass" in hist_props:
        mass = hist_props["mass"]
        del hist_props["mass"]
    else:
        mass = True

    if "mids" in hist_props:
        mids = hist_props["mids"]
        del hist_props["mids"]
    else:
        mids = True

    plotted = []

    if original:
        x_, y_, H = get_histogram2d(
            data[:, 0],
            data[:, 1],
            mids=mids,
            mass=mass,
            avoid_zero_count=avoid_zero_count,
            hist_props=hist_props,
        )

        if free_energy:
            H = get_free_energy(H)

        X, Y = np.meshgrid(x_, y_)
        plotted.append(ax.contourf(X, Y, H, **contour_props))
    else:

        if cluster_map is None:
            cluster_map = {}

        for cluster, cpoints in sorted(cluster_map.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                x_, y_, H = get_histogram2d(
                    data[cpoints, 0],
                    data[cpoints, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props,
                )

                if free_energy:
                    H = get_free_energy(H)

                if cluster == 0:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(ax.contourf(X, Y, H, **contour_noise_props))
                else:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(ax.contourf(X, Y, H, **contour_props))

                if annotate:
                    plotted.append(
                        annotate_points(
                            ax, annotate_pos, data, cpoints, cluster,
                            annotate_props
                        )
                    )

    return plotted


def plot_histogram2d(
        ax,
        data,
        original=True,
        cluster_map=None,
        clusters=None,
        show_props=None,
        show_noise_props=None,
        hist_props=None,
        free_energy=True,
        annotate=False,
        annotate_pos="mean",
        annotate_props=None):

    if show_props is None:
        show_props = {}

    if show_noise_props is None:
        show_noise_props = {}

    if "extent" in show_props:
        del show_props["extent"]

    if hist_props is None:
        hist_props = {}

    if "avoid_zero_count" in hist_props:
        avoid_zero_count = hist_props["avoid_zero_count"]
        del hist_props["avoid_zero_count"]
    else:
        avoid_zero_count = False

    if "mass" in hist_props:
        mass = hist_props["mass"]
        del hist_props["mass"]
    else:
        mass = True

    if "mids" in hist_props:
        mids = hist_props["mids"]
        del hist_props["mids"]
    else:
        mids = True

    plotted = []

    if original:
        x_, y_, H = get_histogram2d(
            data[:, 0],
            data[:, 1],
            mids=mids,
            mass=mass,
            avoid_zero_count=avoid_zero_count,
            hist_props=hist_props,
        )

        if free_energy:
            H = get_free_energy(H)

        plotted.append(ax.imshow(H, extent=(x_, y_), **show_props))
    else:

        if cluster_map is None:
            cluster_map = {}

        for cluster, cpoints in sorted(cluster_map.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                x_, y_, H = get_histogram2d(
                    data[cpoints, 0],
                    data[cpoints, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props,
                )

                if free_energy:
                    H = get_free_energy(H)

                if cluster == 0:
                    plotted.append(
                        ax.imshow(H, extent=(x_, y_), **show_noise_props)
                    )
                else:
                    plotted.append(ax.imshow(H, extent=(x_, y_), **show_props))

                if annotate:
                    plotted.append(
                        annotate_points(
                            ax, annotate_pos, data, cpoints, cluster,
                            annotate_props
                        )
                    )
    return plotted


def annotate_points(ax, annotate_pos, data, points, cluster, annotate_props=None):
    """Add cluster label annotation for given cluster

    Args:
        ax: Matplotlib `Axes` instance to plot on.
        annotate_pos:
            Where to put the cluster number annotation.
            Can be one of:
                * "mean", Use the cluster mean
                * "random", Use a random point of the cluster
                * dict `{1: (x, y), ...}`, Use a specific coordinate
                    tuple for each cluster. Omitted labels will be placed
                    randomly.
        data: The input data.
        points: The point indices for a cluster.
        cluster: The label of the cluster.
        annotate_props: Will be passed to `ax.annotate`
    """
    if annotate_props is None:
        annotate_props = {}

    if isinstance(annotate_pos, dict):
        try:
            xpos, ypos = annotate_pos[cluster]
        except KeyError:
            choosen = random.sample(points, 1)
            xpos = data[choosen, 0]
            ypos = data[choosen, 1]

    elif annotate_pos == "mean":
        xpos = np.mean(data[points, 0])
        ypos = np.mean(data[points, 1])

    elif annotate_pos == "random":
        choosen = random.sample(points, 1)
        xpos = data[choosen, 0]
        ypos = data[choosen, 1]

    else:
        raise ValueError(
            "Keyword argument `annotate_pos` must be "
            'one of "mean", "random", or a dictionary'
        )

    return ax.annotate(f"{cluster}", xy=(xpos, ypos), **annotate_props)


def get_free_energy(H):
    dG = np.inf * np.ones(shape=H.shape)

    nonzero = H.nonzero()
    dG[nonzero] = -np.log(H[nonzero])
    dG[nonzero] -= np.min(dG[nonzero])

    return dG


def get_histogram2d(
        x: Sequence[float],
        y: Sequence[float],
        mids: bool = True,
        mass: bool = True,
        avoid_zero_count: bool = True,
        hist_props: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, ...]:
    """Compute a two-dimensional histogram.

    Taken and modified from :module:`pyemma.plots.`

    Args:
        x: Sample x-coordinates.
        y: Sample y-coordinates.

    Keyword args:
        hist_props: Kwargs passed to `numpy.histogram2d`
        avoid_zero_count: Avoid zero counts by lifting all histogram
            elements to the minimum value before computing the free
            energy.  If False, zero histogram counts yield infinity in
            the free energy.
        mass: Norm the histogram by the total number of counts, so that
            each bin holds the probability mass values where all
            probabilities sum to 1
        mids: Return the mids of the bin edges instead of the actual
            edges

    Returns:
        The x- and y-edges and the data of the computed histogram
    """

    hist_props_defaults = {
        "bins": 100,
    }

    if hist_props is not None:
        hist_props_defaults.update(hist_props)

    z, x_, y_ = np.histogram2d(x, y, **hist_props_defaults)

    if mids:
        x_ = 0.5 * (x_[:-1] + x_[1:])
        y_ = 0.5 * (y_[:-1] + y_[1:])

    if avoid_zero_count:
        z = np.maximum(z, np.min(z[z.nonzero()]))

    if mass:
        z /= float(z.sum())

    return x_, y_, z.T  # transpose to match x/y-directions
