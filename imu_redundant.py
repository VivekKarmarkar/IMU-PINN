# Import libraries
import pandas as pd
import numpy as np

# Load the file
file_path = 'imu_data.log'
with open(file_path, 'r') as file:
    data = file.readlines()

# Locate where the actual data starts
data_start_index = None
for index, line in enumerate(data):
    if not line.startswith('//'):
        data_start_index = index
        break
    
# Read the data into a DataFrame, skipping the initial metadata lines
df = pd.read_csv(file_path, skiprows=data_start_index)

# Investigate timestamps
t_array = np.array(df['SampleTimeFine'].values)
dt_array = 10**-4 * np.diff(t_array)
if all(dt_array > 0):
    print("Timestamps are in the correct order...")
dt_unique_array = np.unique(dt_array)
len_dt_unique = len(dt_unique_array)
if len_dt_unique == 1:
    print("Sampling rate is fixed...")
dt = dt_unique_array.item()
print(f"Sampling interval dt: {dt:.1e} seconds")
print("----------------------------------")

# Check if velocity increments are redundant
vel_x_check_arr = np.abs(np.array(df['Acc_X']*dt - df['VelInc_X']))
vel_x_check_max, vel_x_check_mean, vel_x_check_min = vel_x_check_arr.max(), vel_x_check_arr.mean(), vel_x_check_arr.min()
print(f"vel_x_checks, max: {vel_x_check_max:.1e}, mean: {vel_x_check_mean:.1e}, min: {vel_x_check_min:.1e}")

vel_y_check_arr = np.abs(np.array(df['Acc_Y']*dt - df['VelInc_Y']))
vel_y_check_max, vel_y_check_mean, vel_y_check_min = vel_y_check_arr.max(), vel_y_check_arr.mean(), vel_y_check_arr.min()
print(f"vel_y_checks, max: {vel_y_check_max:.1e}, mean: {vel_y_check_mean:.1e}, min: {vel_y_check_min:.1e}")

vel_z_check_arr = np.abs(np.array(df['Acc_Z']*dt - df['VelInc_Z']))
vel_z_check_max, vel_z_check_mean, vel_z_check_min = vel_z_check_arr.max(), vel_z_check_arr.mean(), vel_z_check_arr.min()
print(f"vel_z_checks, max: {vel_z_check_max:.1e}, mean: {vel_z_check_mean:.1e}, min: {vel_z_check_min:.1e}")

print("Velocity increment data is redundant...")
print("----------------------------------")

# Check if orientation increments are redundant
ori_x_check_arr = np.abs(np.array(df['Gyr_X']*(dt/2) - df['OriInc_q1']))
ori_x_check_max, ori_x_check_mean, ori_x_check_min = ori_x_check_arr.max(), ori_x_check_arr.mean(), ori_x_check_arr.min()
print(f"ori_x_checks, max: {ori_x_check_max:.1e}, mean: {ori_x_check_mean:.1e}, min: {ori_x_check_min:.1e}")

ori_y_check_arr = np.abs(np.array(df['Gyr_Y']*(dt/2) - df['OriInc_q2']))
ori_y_check_max, ori_y_check_mean, ori_y_check_min = ori_y_check_arr.max(), ori_y_check_arr.mean(), ori_y_check_arr.min()
print(f"ori_y_checks, max: {ori_y_check_max:.1e}, mean: {ori_y_check_mean:.1e}, min: {ori_y_check_min:.1e}")

ori_z_check_arr = np.abs(np.array(df['Gyr_Z']*(dt/2) - df['OriInc_q3']))
ori_z_check_max, ori_z_check_mean, ori_z_check_min = ori_z_check_arr.max(), ori_z_check_arr.mean(), ori_z_check_arr.min()
print(f"ori_z_checks, max: {ori_z_check_max:.1e}, mean: {ori_z_check_mean:.1e}, min: {ori_z_check_min:.1e}")

print("Orientation increment data is redundant...")
print("----------------------------------")