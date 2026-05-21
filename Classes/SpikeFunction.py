import numpy as np
import matplotlib.pyplot as plt

class SpikeFunc:
    def __init__(self, x_span:float = 0.75, spike_count: int = 3, amplitude: float = 1.0):
        self.span = x_span
        self.spikes = spike_count
        self.amplitude = amplitude
        self.spike_slope_width = x_span / spike_count  # Fixed: width of each rising/falling segment

    def func(self, x):
        if x <= -self.span or x >= self.span:
            return 0.0

        x_shifted = x + self.span
        segment = int(x_shifted / self.spike_slope_width)
        x_in_segment = x_shifted % self.spike_slope_width

        if segment % 2 == 0:  # Rising
            return (x_in_segment / self.spike_slope_width) * self.amplitude
        else:  # Falling
            return ((self.spike_slope_width - x_in_segment) / self.spike_slope_width) * self.amplitude

    def plot(self, resolution: int = 1000, show_grid: bool = False):
        """Plot the function using matplotlib."""
        x = np.linspace(-self.span*1.25, self.span*1.25, resolution)
        y = np.array([self.func(xi) for xi in x])
        plt.ylim(-0.1 * self.amplitude, 1.1 * self.amplitude)
        plt.xlim(-self.span*1.25, self.span*1.25)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Spike Function (span={self.span}, spikes={self.spikes}, amplitude={self.amplitude})')
        plt.grid(show_grid)
        plt.show()

