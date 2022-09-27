from commonnn._primitive_types import P_AVALUE, P_AVALUE32, P_AINDEX, P_ABOOL


def test_primitive_consistency():
    isinstance(P_AVALUE, float)
    isinstance(P_AVALUE32, float)
    isinstance(P_AINDEX, int)
    isinstance(P_ABOOL, bool)
