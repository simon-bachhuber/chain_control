import numpy as np

from .ring_array import RingArray


def test_indexing():
    for dtype in [np.float32, np.float64]:
        arr = RingArray(maxlen=3, dtype=dtype)
        arr.append(1.0)
        assert (arr[:] == np.array([[1.0]], dtype=dtype)).all()
        arr.append(2.0)
        arr.append(3.0)
        assert (arr[:] == np.array([[1.0], [2.0], [3.0]], dtype=dtype)).all()
        arr.append(4.0)
        assert (arr[:] == np.array([[2.0], [3.0], [4.0]], dtype=dtype)).all()
        assert (arr[1:] == np.array([[3.0], [4.0]], dtype=dtype)).all()
        assert (arr[1:3] == np.array([[3.0], [4.0]], dtype=dtype)).all()
        assert (arr[-1] == np.array([[4.0]], dtype=dtype)).all()
        assert (arr[-2] == np.array([[3.0]], dtype=dtype)).all()
        assert (arr[-2:] == np.array([[3.0], [4.0]], dtype=dtype)).all()
        assert (arr[-2:0] == np.array([[3.0], [4.0]], dtype=dtype)).all()
        assert (arr[-3:-1] == np.array([[2.0], [3.0]], dtype=dtype)).all()

        # TODO
        # assert (arr[:-1] == np.array([[2.0], [3.0]], dtype=dtype)).all()
        # assert (arr[0:-1] == np.array([[2.0], [3.0]], dtype=dtype)).all()
