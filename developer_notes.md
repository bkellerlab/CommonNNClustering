meta dictionary
---------------

Some types have a `_meta` attribute and a corresponding `meta` property. It holds a dictionary of key value pairs used to
store additional type specific information. Information used
at some point are:

* Labels

   * origin
   * params
   * reference

* InputData

   * access_coordinates
   * access_distances
   * access_neighbours

* Bundle


The key and values are not restricted. To save memory if meta information is not used, the `_meta` attribute may be set to `None`.
The `meta` property should always return an empty dictionary in this cases. For extension types, `_meta` should be public typed as `dict`. The corresponding setter for `meta` should check the given value to be either `None` or a `MutableMapping`.
