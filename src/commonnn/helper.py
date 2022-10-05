from collections.abc import MutableMapping


def set_dict_attribute(obj, attr, value):
    if value is None:
        setattr(obj, attr, None)
        return
    if not isinstance(value, MutableMapping):
        raise TypeError("Expected a mutable mapping")
    setattr(obj, attr, value)


def get_dict_attribute(obj, attr):
    value = getattr(obj, attr)
    if value is None:
        return {}
    return value
