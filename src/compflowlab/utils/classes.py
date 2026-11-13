import numpy as np

class MovingAverage:
    def __init__(self, window_size, num_var):
        self.window_size = window_size
        self.window = np.zeros((num_var, window_size))
        self.count = 0
        self.sum = np.zeros(num_var)
        self.avg = np.full_like(self.sum, np.nan)

    def update(self, value):
        if np.any(np.isnan(value)):
            return

        if self.count == self.window_size:
            self.sum -= self.window[:,0]
            self.sum += value

            self.window[:,:-1] = self.window[:,1:]
            self.window[:,-1] = value
        else:
            self.sum += value
            self.window[:,self.count] = value
            self.count += 1

        if self.count < 0.5*self.window_size:
            self.avg = np.full_like(self.sum, np.nan)
        else:
            self.avg = self.sum / self.count

    def reset(self):
        self.window[:,:] = 0.0
        self.count = 0
        self.sum[:] = 0.0
        self.avg[:] = np.nan

class MedianAbsoluteDeviation:
    def __init__(self, window_size, num_var):
        self.window_size = window_size
        self.window = np.zeros((num_var, window_size))
        self.count = 0
        self.next_index = 0
        self.center = np.zeros(num_var)
        self.MAD = np.full_like(self.center, np.nan)

    def update(self, value):
        self.window[:,self.next_index] = value

        self.next_index = (self.next_index + 1) % self.window_size

        self.count = min(self.count + 1, self.window_size)

        active = self.window[:,:self.count]

        self.center = np.median(active, axis=1)
        res = active - self.center[:, None]
        res_median = np.median(np.abs(res), axis=1)

        if self.count < 0.5*self.window_size:
            self.MAD = np.full_like(self.center, np.nan)
            self.center[:] = np.nan
        else:
            self.MAD = res_median / np.abs(self.center)

    def reset(self):
        self.window[:,:] = 0.0
        self.count = 0
        self.next_index = 0
        self.center[:] = np.nan
        self.MAD[:] = np.nan
