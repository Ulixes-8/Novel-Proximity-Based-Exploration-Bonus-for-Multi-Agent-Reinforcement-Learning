import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# plt.style.use('seaborn-poster') # Use a predefined style for larger fonts and figures
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
# plt.title('Matplotlib Example')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.legend()
# plt.grid(True)

import seaborn as sns
import numpy as np

# Sample data
data = np.random.rand(10, 12)

sns.set_context('paper') # Set the context to "paper" for finer control
plt.figure(figsize=(8, 6))
ax = sns.heatmap(data, linewidth=0.5, cmap='coolwarm')
ax.set_title('Seaborn Heatmap Example')
plt.savefig('matplotlib_example.png') # Save in PDF format for high quality
