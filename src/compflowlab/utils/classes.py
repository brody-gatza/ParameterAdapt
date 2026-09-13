import numpy as np

class MovingAverage:
    def __init__(self, window_size, num_var):
        self.window_size = window_size
        self.window = np.zeros((num_var, window_size))
        self.count = 0
        self.sum = np.zeros(num_var)
        self.avg = np.zeros(num_var)

    def update(self, value):
        if self.count == self.window_size:
            self.sum -= self.window[:,0]
            self.sum += value

            self.window[:,:-1] = self.window[:,1:]
            self.window[:,-1] = value
        else:
            self.sum += value
            self.window[:,self.count] = value
            self.count += 1

        self.avg = self.sum / self.count

    def reset(self):
        self.window[:,:] = 0.0
        self.count = 0
        self.sum[:] = 0.0
        self.avg[:] = 0.0

class MedianAbsoluteDeviation:
    def __init__(self, window_size, num_var):
        self.window_size = window_size
        self.window = np.zeros((num_var, window_size))
        self.count = 0
        self.next_index = 0
        self.center = np.zeros(num_var)
        self.MAD = np.zeros(num_var)

    def update(self, value):
        self.window[:,self.next_index] = value

        self.next_index = (self.next_index + 1) % self.window_size

        self.count = min(self.count + 1, self.window_size)

        active = self.window[:,:self.count]

        self.center = np.median(active, axis=1)
        res = active - self.center[:, None]
        res_median = np.median(np.abs(res), axis=1)
        self.MAD = res_median / np.abs(self.center)

    def reset(self):
        self.window[:,:] = 0.0
        self.count = 0
        self.next_index = 0
        self.center[:] = 0.0
        self.MAD[:] = 0.0
