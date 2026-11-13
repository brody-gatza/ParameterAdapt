import numpy as np


class MovingAverage:
    def __init__(self, window_size, num_var):
        self.window_size = window_size
        self.window = np.zeros((num_var, 0))
        self.iterations = []
        self.count = 0
        self.sum = np.zeros(num_var)
        self.avg = np.full_like(self.sum, np.nan)

    def update(self, iteration, value):
        if np.any(np.isnan(value)):
            return

        while (
            self.iterations
            and self.iterations[0] + self.window_size < iteration
        ):
            
            self.sum -= self.window[:, 0]
            self.window = self.window[:, 1:]
            self.iterations.pop(0)

        self.window = np.column_stack((self.window, value))
        self.iterations.append(iteration)
        self.sum += value
        self.count = len(self.iterations)

        # if iteration - self.iterations[0] < 0.5 * self.window_size:
        #     self.avg = np.full_like(self.sum, np.nan)
        # else:
        #     self.avg = self.sum / self.count
        self.avg = self.sum / self.count

    def reset(self):
        self.window = self.window[:, :0]
        self.iterations.clear()
        self.count = 0
        self.sum[:] = 0.0
        self.avg[:] = np.nan


class MedianAbsoluteDeviation:
    def __init__(self, window_size, num_var):
        self.window_size = window_size
        self.window = np.zeros((num_var, 0))
        self.iterations = []
        self.count = 0
        self.center = np.zeros(num_var)
        self.MAD = np.full_like(self.center, np.nan)

    def update(self, iteration, value):
        while (
            self.iterations
            and self.iterations[0] + self.window_size < iteration
        ):
            self.window = self.window[:, 1:]
            self.iterations.pop(0)

        self.window = np.column_stack((self.window, value))
        self.iterations.append(iteration)
        self.count = len(self.iterations)

        self.center = np.median(self.window, axis=1)
        res = self.window - self.center[:, None]
        res_median = np.median(np.abs(res), axis=1)

        # if iteration - self.iterations[0] < 0.5 * self.window_size:
        #     self.MAD = np.full_like(self.center, np.nan)
        #     self.center[:] = np.nan
        # else:
        #     self.MAD = res_median / np.abs(self.center)
        self.MAD = res_median / np.abs(self.center)

    def reset(self):
        self.window = self.window[:, :0]
        self.iterations.clear()
        self.count = 0
        self.center[:] = np.nan
        self.MAD[:] = np.nan