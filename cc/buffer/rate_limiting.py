from abc import ABC, abstractmethod
from collections import deque


class AbstractRateLimiter(ABC):

    @abstractmethod
    def count_sample_up(self) -> None:
        pass 

    @abstractmethod
    def count_insert_up(self) -> None:
        pass 

    @abstractmethod
    def sample_block(self) -> bool:
        pass 

    @abstractmethod
    def insert_block(self) -> bool:
        pass 


class NoRateLimitingLimiter(AbstractRateLimiter):

    def count_sample_up(self):
        pass 

    def count_insert_up(self):
        pass 

    def sample_block(self) -> bool:
        return False

    def insert_block(self) -> bool:
        return False


class RateLimiter(AbstractRateLimiter):
    def __init__(self, maxlen=1_000, target_ratio=0.1, error_margin=0.01, update_ratio_freq = 10):
        self.deque = deque(maxlen=maxlen) 
        self.maxlen = maxlen
        self.target_ratio = target_ratio
        self.error_margin = error_margin
        self.update_ratio_freq = update_ratio_freq 
        self._count = 0

        # there is no blocking in the beginning
        self._last_ratio = target_ratio

        # because of division 
        self.count_insert_up()
        self.count_sample_up()

    def _count_up(self):
        self._count += 1

    def count_sample_up(self):
        self._count_up()
        self.deque.append(1)
    
    def count_insert_up(self):
        self._count_up()
        self.deque.append(0)

    @property
    def _ratio(self) -> float:
        if self._count > self.update_ratio_freq:
            self._count = 0 
            self._last_ratio = self._current_ratio()
        
        return self._last_ratio 

    def _current_ratio(self):
        s = self.deque.count(1)
        if self.deque.maxlen == len(self.deque):
            i = self.maxlen - s 
        else:
            i = self.deque.count(0)
        return s/i 

    def sample_block(self) -> bool:
        """Return `True` if too many samples

        Returns:
            bool: 
        """
        if self._ratio > self.target_ratio+self.error_margin:
            return True 
        else:
            return False 
    
    def insert_block(self) -> bool:
        if self._ratio < self.target_ratio-self.error_margin:
            return True 
        else:
            return False