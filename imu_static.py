# Import libraries
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Load the reference file
file_path_ref = 'ori_reference.log'
with open(file_path_ref, 'r') as file:
    lines = file.readlines()

# Filter out lines from the reference file that start with "//"
filtered_lines = [line for line in lines if not line.strip().startswith('//')]

# Write the filtered lines to a temporary CSV file
temp_file_path = 'filtered_data.csv'
with open(temp_file_path, 'w') as file:
    file.writelines(filtered_lines)

# Read the CSV file and specify that the first row contains the headers
df_ref = pd.read_csv(temp_file_path, header=1)

# Extract and normalize reference timestamps
t_ref_raw_array = jnp.array(df_ref['SampleTimeFine'].values).reshape(-1, 1)
t_ref_array_full = 10**-4 * (t_ref_raw_array - t_ref_raw_array[0])

# Extract reference orientation data
data_quat_full = jnp.array(df_ref[list(df_ref.columns[1:])].values.tolist())

# Function to convert quaternion to roll, pitch, yaw
def quat_to_rpy(quat):
    w, x, y, z = quat
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jnp.arcsin(2.0 * (w * y - z * x))
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.array([roll, pitch, yaw])

# Vectorized conversion
rpy_angles_rad = jax.vmap(quat_to_rpy)(data_quat_full)

# Convert the angles from radians to degrees
rpy_angles_deg = jnp.degrees(rpy_angles_rad)

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

# Normalize timestamps
t_array = np.array(df['SampleTimeFine'].values)
t_normalized_array = 10**-4 *(t_array - t_array[0])

# Extract accelerometer and gyroscope data
data_acc = np.array(df[['Acc_X', 'Acc_Y', 'Acc_Z']].values.tolist())
data_gyro = np.array(df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values.tolist())

# Extract movement trials
print("Extracting movement trials...")

# List to store the start and end indices of movement trials
start_idx_list = []
end_idx_list = []

# Parameters
window_size = 500
error_size = 2*window_size
std_threshold = 1.0
constant_jump = 9.81
threshold_jump = 10**-1

# Function to compute the standard deviation of a window
def window_std(data, start_idx, window_size):
    return np.std(data[start_idx:start_idx + window_size])

data_jump = data_acc[:,2]

current_state = "start"
for i in range(1, len(data_acc) - window_size - 1):
    current_std = window_std(data_acc[:, 2], i, window_size)
    left_std = window_std(data_acc[:, 2], i - window_size, window_size)
    right_std = window_std(data_acc[:, 2], i + window_size, window_size)
    
    current_std = window_std(data_jump, i, window_size)
    left_std = window_std(data_jump, i - window_size, window_size)
    right_std = window_std(data_jump, i + window_size, window_size)
    
    if current_state == "start":
        
        if current_std > std_threshold and left_std < std_threshold and right_std > std_threshold:
            
            bool_start_zero = np.abs(np.mean(np.abs(data_acc[i-window_size:i, 0])) - constant_jump) < threshold_jump
            bool_start_one = np.abs(np.mean(np.abs(data_acc[i-window_size:i, 1])) - constant_jump) < threshold_jump
            bool_start_two = np.abs(np.mean(np.abs(data_acc[i-window_size:i, 2])) - constant_jump) < threshold_jump
            
            if bool_start_zero or bool_start_one or bool_start_two:
                start_idx = i - error_size
                start_idx_list.append(start_idx)
                current_state = "end"
                
    else:
        
        if current_std < std_threshold and left_std > std_threshold and right_std < std_threshold:
            
            bool_end_zero = np.abs(np.mean(np.abs(data_acc[i:i+window_size, 0])) - constant_jump) < threshold_jump
            bool_end_one = np.abs(np.mean(np.abs(data_acc[i:i+window_size, 1])) - constant_jump) < threshold_jump
            bool_end_two = np.abs(np.mean(np.abs(data_acc[i:i+window_size, 2])) - constant_jump) < threshold_jump
            
            if bool_end_zero or bool_end_one or bool_end_two:
                end_idx = i + error_size
                end_idx_list.append(end_idx)
                current_state = "start"

# Output the result
print("Start indices of movement trials:", start_idx_list)
print("End indices of movement trials:", end_idx_list)
print("----------------------------------")

# Manual correction
end_idx_list[9] += 2000
end_idx_list[10] += 2000
end_idx_list.pop(0)
start_idx_list.pop(1)
end_idx_list[0] += 2000

# Define arrays for orientation estimates
roll_estimates = np.full(t_normalized_array.shape, np.nan)
pitch_estimates = np.full(t_normalized_array.shape, np.nan)
yaw_estimates = np.full(t_normalized_array.shape, np.nan)

roll_estimates[0] = 0
pitch_estimates[0] = 0
yaw_estimates[0] = 0

est_end_idx_list = [0] + end_idx_list
est_start_idx_list = start_idx_list + [len(t_normalized_array)-1]

# Populate orientation estimate arrays during static trials
for (e,s) in zip(est_end_idx_list, est_start_idx_list):
    yaw_estimates[e+1:s] = 0
    if np.abs(np.mean(data_acc[e:s,1]) - 9.81) < 0.3:
        roll_estimates[e+1:s] = 90
        pitch_estimates[e+1:s] = 0
    elif np.abs(np.mean(data_acc[e:s,1]) + 9.81) < 0.3:
        roll_estimates[e+1:s] = -90
        pitch_estimates[e+1:s] = 0
    elif np.abs(np.mean(data_acc[e:s,0]) - 9.81) < 0.3:
        roll_estimates[e+1:s] = 0
        pitch_estimates[e+1:s] = -90
    elif np.abs(np.mean(data_acc[e:s,0]) + 9.81) < 0.3:
        roll_estimates[e+1:s] = 0
        pitch_estimates[e+1:s] = 90
    else:
        roll_estimates[e+1:s] = 0
        pitch_estimates[e+1:s] = 0
        
# Plot roll
plt.figure()
plt.plot(t_normalized_array, roll_estimates, color='b', linewidth=3, label='roll estimates')
plt.plot(t_ref_array_full, rpy_angles_deg[:,0], 'r', label='roll reference')
plt.axhline(y=180, color='k', linestyle='--')
plt.axhline(y=90, color='k', linestyle='--')
plt.axhline(y=-90, color='k', linestyle='--')
plt.axhline(y=-180, color='k', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
for (e,s) in zip(est_end_idx_list, est_start_idx_list):
    x1_val = t_normalized_array[e]
    x2_val = t_normalized_array[s]
    plt.axvline(x=x1_val, color='m', linestyle='--')
    plt.axvline(x=x2_val, color='g', linestyle='--')
    plt.fill_betweenx(y=[-180, 180], x1=x1_val, x2=x2_val, color='grey', alpha=0.5)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
plt.xlabel("time (s)")
plt.ylabel("roll (degrees)")
plt.title("Roll (static portions)")

# Plot pitch
plt.figure()
plt.plot(t_normalized_array, pitch_estimates, color='b', linewidth=3, label='pitch estimates')
plt.plot(t_ref_array_full, rpy_angles_deg[:,1], 'r', label='pitch reference')
plt.axhline(y=180, color='k', linestyle='--')
plt.axhline(y=90, color='k', linestyle='--')
plt.axhline(y=-90, color='k', linestyle='--')
plt.axhline(y=-180, color='k', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
for (e,s) in zip(est_end_idx_list, est_start_idx_list):
    x1_val = t_normalized_array[e]
    x2_val = t_normalized_array[s]
    plt.axvline(x=x1_val, color='m', linestyle='--')
    plt.axvline(x=x2_val, color='g', linestyle='--')
    plt.fill_betweenx(y=[-180, 180], x1=x1_val, x2=x2_val, color='grey', alpha=0.5)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
plt.xlabel("time (s)")
plt.ylabel("pitch (degrees)")
plt.title("Pitch (static portions)")

# Plot yaw
plt.figure()
plt.plot(t_normalized_array, yaw_estimates, color='b', linewidth=3, label='yaw estimates')
plt.plot(t_ref_array_full, rpy_angles_deg[:,2], 'r', label='yaw reference')
plt.axhline(y=180, color='k', linestyle='--')
plt.axhline(y=90, color='k', linestyle='--')
plt.axhline(y=-90, color='k', linestyle='--')
plt.axhline(y=-180, color='k', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
for (e,s) in zip(est_end_idx_list, est_start_idx_list):
    x1_val = t_normalized_array[e]
    x2_val = t_normalized_array[s]
    plt.axvline(x=x1_val, color='m', linestyle='--')
    plt.axvline(x=x2_val, color='g', linestyle='--')
    plt.fill_betweenx(y=[-180, 180], x1=x1_val, x2=x2_val, color='grey', alpha=0.5)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
plt.xlabel("time (s)")
plt.ylabel("yaw (degrees)")
plt.title("Yaw (static portions)")