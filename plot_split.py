import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
# Adjust path if necessary based on where the script is run
# Assuming running from C:/Users/Scent/Project/課題
file_path = 'BikeSharing/data/train.csv'
if not os.path.exists(file_path):
    # Fallback if running from inside BikeSharing folder
    file_path = 'data/train.csv'

df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by datetime just in case
df = df.sort_values('datetime').reset_index(drop=True)

# Split logic matching main.py
# 1. Split into Train+Val (80%) and Test (20%)
n_total = len(df)
n_test = int(n_total * 0.2)
n_train_val = n_total - n_test

# 2. Split Train+Val into Train (75% of 80% = 60%) and Val (25% of 80% = 20%)
n_val = int(n_train_val * 0.25)
n_train = n_train_val - n_val

# Create segments
train_df = df.iloc[:n_train]
val_df = df.iloc[n_train:n_train+n_val]
test_df = df.iloc[n_train+n_val:]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train_df['datetime'], train_df['count'], label='Train (60%)', color='#1f77b4', alpha=0.7, linewidth=1)
plt.plot(val_df['datetime'], val_df['count'], label='Validation (20%)', color='#ff7f0e', alpha=0.7, linewidth=1)
plt.plot(test_df['datetime'], test_df['count'], label='Test (20%)', color='#2ca02c', alpha=0.7, linewidth=1)

plt.title('Time Series Data Split Strategy', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Rental Count', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
output_dir = 'BikeSharing/output'
if not os.path.exists(output_dir):
    output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'data_split.png')
plt.savefig(output_path, dpi=300)
print(f"Saved plot to {output_path}")
