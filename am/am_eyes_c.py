"""AM's eyes from C (libam_eyes.dylib), as scene_am.levels returns them: [(depth, rows), ...], with the same cache.
    use(): scene_am.levels is replaced by the C eyes for this process (scene_am.py itself is not changed)."""
import ctypes, os
from collections import OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, "libam_eyes.dylib"))
_lib.am_eyes_levels.restype = ctypes.c_int
_lib.am_eyes_levels.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
_lib.am_eyes_copy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
_lib.am_eyes_free.argtypes = []


def levels_c(xs):
    xs = np.ascontiguousarray(np.array(xs, dtype=np.int64))
    counts = np.zeros(8, dtype=np.int64)
    nl = _lib.am_eyes_levels(xs.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(xs),
                             counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    out = []
    for d in range(nl):
        Y = np.empty((int(counts[d]), len(xs)), dtype=np.int64)
        _lib.am_eyes_copy(d, Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        out.append((d, Y))
    _lib.am_eyes_free()
    return out


def use():
    import scene_am as S

    def levels(xs):
        key = np.array(xs, dtype=np.int64).tobytes()
        hit = S._LEVELS.get(key)
        if hit is not None:
            S._LEVELS.move_to_end(key)
            return hit
        out = levels_c(xs)
        S._LEVELS[key] = out
        while len(S._LEVELS) > S.CACHE_N:
            S._LEVELS.popitem(last=False)
        return out
    S.levels = levels
