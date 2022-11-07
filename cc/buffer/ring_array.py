from typing import Union

import numpy as np


class RingArray:
    def __init__(self, maxlen: int, dtype=np.float32, shape=(1,)):
        self._i_start = 0
        self.maxlen = maxlen
        self._arr = np.zeros((maxlen, *shape), dtype=dtype)
        self._curr_len = 0

    def __len__(self):
        return self._curr_len

    def append(self, ele):
        self._curr_len = min(self.maxlen, self._curr_len + 1)
        self._arr[self._i_start] = ele
        self._i_start = (self._i_start + 1) % self.maxlen

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            if key.start is None:
                start = 0
            else:
                start = key.start
            if key.stop is None:
                if start < 0:
                    stop = 0
                else:
                    stop = self._curr_len
            else:
                stop = key.stop
        else:
            start = key
            stop = start + 1

        start_right = start + self._i_start
        stop_right = start_right + (stop - start)
        stop_right = min(stop_right, self._curr_len)
        start_left = 0
        stop_left = min((stop - start) - (stop_right - start_right), self._i_start)

        r = np.r_[start_right:stop_right, start_left:stop_left]
        return self._arr[r]
