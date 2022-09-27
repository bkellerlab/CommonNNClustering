cdef class ClusterParameters:
    """Input parameters for clustering procedure"""

    def __cinit__(
            self,
            radius_cutoff: float,
            similarity_cutoff: int = 0,
            similarity_cutoff_continuous: float = 0.,
            n_member_cutoff: int = None,
            current_start: int = 1):

        if n_member_cutoff is None:
            n_member_cutoff = similarity_cutoff

        self.radius_cutoff = radius_cutoff
        self.similarity_cutoff = similarity_cutoff
        self.similarity_cutoff_continuous = similarity_cutoff_continuous
        self.n_member_cutoff = n_member_cutoff
        self.current_start = current_start

    def __init__(
            self,
            radius_cutoff: float,
            similarity_cutoff: int = 0,
            similarity_cutoff_continuous: float = 0.,
            n_member_cutoff: int = None,
            current_start: int = 1):
        """

        Args:
            radius_cutoff: Neighbour search radius :math:`r`.

        Keyword args:
            similarity_cutoff:
                Value used to check the similarity criterion, i.e. the
                minimum required number of shared neighbours :math:`n_\mathrm{c}`.
            similarity_cutoff_continuous:
                Same as `similarity_cutoff` but allowed to be a floating point
                value.
            n_member_cutoff:
                Minimum required number of points in neighbour lists
                to be considered for a similarity check
                (used for example in :obj:`~cnnclustering._types.Neighbours.enough`).
                If `None`, will be set to `similarity_cutoff`.
            current_start: Use this as the first label for identified clusters.
        """
        pass

    def to_dict(self):
        """Return a Python dictionary of cluster parameter key-value pairs"""

        return {
            "radius_cutoff": self.radius_cutoff,
            "similarity_cutoff": self.similarity_cutoff,
            "similarity_cutoff_continuous": self.similarity_cutoff_continuous,
            "n_member_cutoff": self.n_member_cutoff,
            "current_start": self.current_start,
            }

    def __repr__(self):
        return f"{self.to_dict()!r}"

    def __str__(self):
        return f"{self.to_dict()!s}"
