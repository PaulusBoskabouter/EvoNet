import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

    def plot(self, resolution: int = 1000, show_grid: bool = False, output_data:tuple=None, plot_title:str=None, save_path:Path = None):
        """Plot the function using matplotlib."""

        plt.figure()
        x = np.linspace(-self.span*1.25, self.span*1.25, resolution)
        y = np.array([self.func(xi) for xi in x])
        

        plt.ylim(-0.1 * self.amplitude, 1.1 * self.amplitude)
        plt.xlim(-self.span*1.25, self.span*1.25)
        plt.plot(x, y, linewidth='1.5', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('y')
        if output_data is not None:
            for x_out, y_out, label in output_data: 
                plt.plot(x_out, y_out, '--', label=label, alpha=0.98)
        
            if len(output_data)> 1:
                plt.legend()
        if plot_title is None:
            plt.title(f'Spike Function (span={self.span}, spikes={self.spikes}, amplitude={self.amplitude})')
        else:
            plt.title(plot_title)
        plt.grid(show_grid)
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
    

    def mse(self, X, Y_predict):
        # actual_y = self.func(X)
        actual_y = np.array([self.func(xi) for xi in X])
        return np.mean((actual_y - Y_predict) **2 )