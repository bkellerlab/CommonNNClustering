import importlib
import sys


cython_test_modules = ["test_primitive"]

for mod in cython_test_modules:
    print(mod)
    try:
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                setattr(sys.modules[__name__], name, item)
    except ImportError:
        print(f"Cannot import module {mod}")
        pass
