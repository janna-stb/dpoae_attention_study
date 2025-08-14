import numpy as np

class RunningStatsArray:
    """Class to compute running mean for complex arrays."""
    
    def __init__(self):
        self.n = 0
        self.mean = None  

    def push(self, x):
        """Update the running mean with a new array x."""

        x = np.asarray(x)
        if self.mean is None:
            self.mean = np.zeros_like(x)

        self.n += 1
        old_weight = (self.n - 1) / self.n
        new_weight = 1 / self.n

        self.mean = old_weight * self.mean + new_weight * x