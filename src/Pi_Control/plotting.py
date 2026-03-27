import matplotlib.pyplot as plt
import pandas as pd
import os

# Define base path and file locations
base_path = os.path.dirname(os.path.abspath(__file__))
pan_error_file = 'runs/run_2025-03-07_01-13-48/data_pan.csv'
pan_path = os.path.join(base_path, pan_error_file)

# Load CSV data
df = pd.read_csv(pan_path)

# Print first row to verify data
print("Pan Error file loaded. First row:")
print(df.iloc[0])

# Apply time filter for analysis
df_filtered = df[(df["Time"] > 15) & (df["Time"] < 22)]
df_filtered = df

# Define PID values (assuming `config` dictionary exists)
config = {
    "kp_pan": 0.5,  # Example values
    "ki_pan": 0.0,
    "kd_pan": 0.5,
    "kp_tilt": 0.3,
    "ki_tilt": 0,
    "kd_tilt": 0.3,
}

# Plot Pan Error
plt.figure(figsize=(15, 6))
plt.plot(df_filtered["Time"], df_filtered["Pan_Error"], linestyle="-", label="Pan Error", color="b")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.title(f"Tracking Error Over Time - Pan Error\n"
          f"PID Kp: {config['kp_pan']}, Ki: {config['ki_pan']}, Kd: {config['kd_pan']}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pan_error.png")
plt.show()

# Plot Tilt Error
plt.figure(figsize=(12, 6))
plt.plot(df_filtered["Time"], df_filtered["Tilt_Error"], linestyle="-", label="Tilt Error", color="r")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.title(f"Tracking Error Over Time - Tilt Error\n"
          f"PID Kp: {config['kp_tilt']}, Ki: {config['ki_tilt']}, Kd: {config['kd_tilt']}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tilt_error.png")
plt.show()
