import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Sample signal: sin(t)
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]  # Time step
signal = np.sin(t)

# Method 1: Riemann integration (simple sum)
integral_riemann = np.cumsum(signal) * dt

# Method 2: Trapezoidal integration (more accurate)
integral_trapz = cumulative_trapezoid(signal, t, initial=0)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal (sin)', color='blue')
plt.ylabel("Amplitude")
plt.title("Signal and Its Integral")

plt.subplot(2, 1, 2)
plt.plot(t, integral_riemann, label='Riemann Integral', linestyle='--', color='green')
plt.plot(t, integral_trapz, label='Trapezoidal Integral', linestyle='-', color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Integrated Value")
plt.legend()

plt.tight_layout()
plt.show()
