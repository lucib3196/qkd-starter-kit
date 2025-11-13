import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define file paths and their labels
file_paths = {
    "Main": "/home/lberm/Documents/github/Senior_Desing_QKD_2024/data_main.csv",
    "Pan": "/home/lberm/Documents/github/Senior_Desing_QKD_2024/data_pan.csv",
    "Tilt": "/home/lberm/Documents/github/Senior_Desing_QKD_2024/data_tilt.csv",
    "Calibration": "/home/lberm/Documents/github/Senior_Desing_QKD_2024/data_calib.csv"
}

for label, path in file_paths.items():
    try:
        # Attempt to read the CSV file into a DataFrame
        df = pd.read_csv(path)
        print(f"{label} file loaded. First row:")
        print(df.iloc[0])
        
        # Create a new figure for this file
        plt.figure(figsize=(10, 8))
        
        # Plot the error data.
        # If the file has a generic "Error" column, use that.
        # Otherwise, check for "Pan_Error" and/or "Tilt_Error".
        if "Error" in df.columns:
            plt.plot(df["Time"], df["Error"], linestyle="-", label="Error")
        else:
            if "Pan_Error" in df.columns:
                plt.plot(df["Time"], df["Pan_Error"], linestyle="-", label="Pan Error")
            if "Tilt_Error" in df.columns:
                plt.plot(df["Time"], df["Tilt_Error"], linestyle="--", label="Tilt Error")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Error (deg)")
        plt.title(f"Tracking Error Over Time - {label}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not load {label} file at {path}. Error: {e}")
