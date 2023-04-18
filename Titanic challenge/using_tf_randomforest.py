import matplotlib.pyplot as plt
import numpy as np

# create some sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)

# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2)

# plot the data on the subplots and add labels
axs[0, 0].plot(x, y1)
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('sin(x)')
axs[0, 1].plot(x, y2)
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('cos(x)')
axs[1, 0].plot(x, y3)
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('tan(x)')
axs[1, 1].plot(x, y4)
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('exp(x)')

# set titles for the subplots and the figure
axs[0, 0].set_title('Sine')
axs[0, 1].set_title('Cosine')
axs[1, 0].set_title('Tangent')
axs[1, 1].set_title('Exponential')
fig.suptitle('Plots of Trigonometric and Exponential Functions')

# adjust the layout and display the figure
fig.tight_layout()
plt.show()
