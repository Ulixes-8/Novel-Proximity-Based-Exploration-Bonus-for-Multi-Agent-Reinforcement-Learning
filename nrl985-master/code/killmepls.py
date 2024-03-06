import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulate some reward data
np.random.seed(0)  # For reproducibility
episode_nums = np.arange(1, 101)
rewards = np.random.normal(loc=0.0, scale=1.0, size=100).cumsum() / 10  # Simulated improvement over time

# Calculate the rolling average for a smooth line
rolling_avg_rewards = pd.Series(rewards).rolling(window=10, min_periods=1).mean()

# Convert the rolling average Series to a NumPy array explicitly
rolling_avg_rewards_array = rolling_avg_rewards.to_numpy()

# Now, use the NumPy array for plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Episode Number')
ax.set_ylabel('Average Evaluation Reward')
ax.grid(True)

# Plot the rolling average line using the NumPy array
ax.plot(episode_nums, rolling_avg_rewards_array, label='Smoothed Average Reward', color='blue', lw=2)

# Create a shaded region around the rolling average using the NumPy array for rewards and rolling average
ax.fill_between(episode_nums, rolling_avg_rewards_array, rewards, color='blue', alpha=0.2)
ax.fill_between(episode_nums, rewards, rolling_avg_rewards_array, color='blue', alpha=0.2)

ax.legend()

    # Save the plot
random_number = 234
filename = f'saved_data/figs/test_time_rewards_{random_number}.png'
print(f"Figure saved as {filename}")
plt.savefig(filename)
