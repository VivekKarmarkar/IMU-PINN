# Import libraries
import time
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
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

# Read the CSV file and specify that the second row contains the headers
df_ref = pd.read_csv(temp_file_path, header=1)

# Load the data file
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

# Truncate indices
content_str = "low_freq_trend"
example_idx = 0

truncate_start_idx = None
if content_str == "low_freq_trend":
    truncate_start_idx_list = [8000, 10000, 11000, 12000]
elif content_str == "high_freq_trend":
    truncate_start_idx_list = [90837, 110726, 262387, 220502]
    
truncate_length = 1000

truncate_start_idx = truncate_start_idx_list[example_idx]
truncate_start_ref_idx = None
while truncate_start_ref_idx is None:
    if df['SampleTimeFine'][truncate_start_idx] in df_ref['SampleTimeFine'].values:
        truncate_start_ref_idx = df_ref[df_ref['SampleTimeFine'] == df['SampleTimeFine'][truncate_start_idx]].index[0]
    else:
        truncate_start_idx += 1  

truncate_end_idx = truncate_start_idx + truncate_length
truncate_end_ref_idx = None
while truncate_end_ref_idx is None:
    if df['SampleTimeFine'][truncate_end_idx] in df_ref['SampleTimeFine'].values:
        truncate_end_ref_idx = df_ref[df_ref['SampleTimeFine'] == df['SampleTimeFine'][truncate_end_idx]].index[0]
    else:
        truncate_end_idx += 1  

# Extract and normalize timestamps
t_raw_array = jnp.array(df['SampleTimeFine'].values).reshape(-1, 1)
t_array_full = 10**-4 * (t_raw_array - t_raw_array[0])
t_array = t_array_full[truncate_start_idx:truncate_end_idx, :]

# Extract and normalize reference timestamps
t_ref_raw_array = jnp.array(df_ref['SampleTimeFine'].values).reshape(-1, 1)
t_ref_array_full = 10**-4 * (t_ref_raw_array - t_ref_raw_array[0])
t_ref_array = t_ref_array_full[truncate_start_ref_idx:truncate_end_ref_idx, :]

# Extract accelerometer and gyroscope data
data_acc_full = jnp.array(df[['Acc_X', 'Acc_Y', 'Acc_Z']].values.tolist())
data_gyro_full = jnp.array(df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values.tolist())

# Truncate accelerometer and gyroscope data
data_acc = data_acc_full[truncate_start_idx:truncate_end_idx, :]
data_gyro = data_gyro_full[truncate_start_idx:truncate_end_idx, :]

# Extract reference orientation data
data_quat_full = jnp.array(df_ref[list(df_ref.columns[1:])].values.tolist())

# Truncate reference orientation data
data_quat = data_quat_full[truncate_start_ref_idx:truncate_end_ref_idx, :]

# Assign the boundary condition orientation
q_bc_left = data_quat[0]
q_bc_right = data_quat[-1]

# Define Neural Network class
class PositionQuaternionNN(eqx.Module):
    layers: list
    final_layer_r: eqx.Module
    final_layer_theta: eqx.Module
    final_layer_v: eqx.Module

    def __init__(self, key, hidden_dim=128, hidden_num=4):
        # Create random keys for initializing weights
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        
        self.layers = []
        self.layers.append(eqx.nn.Linear(1, hidden_dim, key=key1))
        for hidden_idx in range(hidden_num):
            key2, subkey = jax.random.split(key2)
            self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=key2))
        
        # Define the final layers for r(t), theta, and v
        self.final_layer_r = eqx.nn.Linear(hidden_dim, 3, key=key3)
        self.final_layer_theta = eqx.nn.Linear(hidden_dim, 1, key=key4)
        self.final_layer_v = eqx.nn.Linear(hidden_dim, 3, key=key5)
        
    def __call__(self, t):
        # Forward pass through shared layers
        x = t
        for layer in self.layers:
            x = jax.nn.sigmoid(layer(x))
        
        # Compute r(t)
        r_t = self.final_layer_r(x)
        
        # Compute theta
        theta = self.final_layer_theta(x)
        
        # Compute v (unit vector part of the quaternion)
        v = self.final_layer_v(x)
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)  # Normalize to ensure it's a unit vector
        
        # Form the quaternion q(t) = [cos(theta), v * jnp.sin(theta)]
        scalar_part = jnp.cos(theta)
        vector_part = v * jnp.sin(theta)
        q_t = jnp.concatenate([scalar_part, vector_part], axis=-1)
        
        return r_t, q_t

# Define quaternion operation from scratch for compatibility with jax framework
def quaternion_scalar_and_vector(q):
    scalar_part = q[..., 0]
    vector_part = q[..., 1:]
    return scalar_part, vector_part

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion.
    
    Args:
        q (jax.numpy.ndarray): Quaternion of shape (..., 4) where the first element is the scalar part 
                               and the remaining three elements are the vector part.
    
    Returns:
        jax.numpy.ndarray: The conjugate of the input quaternion.
    """
    
    scalar_part, vector_part = quaternion_scalar_and_vector(q)
    conjugate = jnp.concatenate([scalar_part[..., jnp.newaxis], -vector_part], axis=-1)
    return conjugate

def quaternion_product(q1, q2):
    """Compute the product of two quaternions.
    
    Args:
        q1, q2 (jax.numpy.ndarray): Quaternions of shape (..., 4) where the first element is the scalar part 
                                    and the remaining three elements are the vector part.
    
    Returns:
        jax.numpy.ndarray: The product of the two input quaternions.
    """
    
    q1 = q1.squeeze()
    q2 = q2.squeeze()
    
    s1, v1 = quaternion_scalar_and_vector(q1)
    s2, v2 = quaternion_scalar_and_vector(q2)
    
    scalar_part = s1 * s2 - jnp.sum(v1 * v2, axis=-1)
    vector_part = s1[..., jnp.newaxis] * v2 + s2[..., jnp.newaxis] * v1 + jnp.cross(v1, v2)
    
    return jnp.concatenate([scalar_part[..., jnp.newaxis], vector_part], axis=-1)

# Define the physical law for gyroscope
def gyroscope_model(pose, t):
    
    dq_dt = jax.jacrev(lambda t: pose(t)[1])
    q_dot = dq_dt(t)
    
    _, q_t = pose(t)
    q_t_conjugate = quaternion_conjugate(q_t)
    
    q_omega = quaternion_product(q_t_conjugate, q_dot)
    _, vec_omega = quaternion_scalar_and_vector(q_omega)
    omega = 2 * vec_omega
    
    return omega

# Define the physical law for accelerometer
def accelerometer_model(pose, t):
    
    r_t, q_t = pose(t)
    q_t_conjugate = quaternion_conjugate(q_t)
    
    d2r_dt2 = jax.jacrev(jax.jacrev(lambda t: pose(t)[0]))
    gravity_acc = jnp.array([0, 0, -9.81])
    
    vec_acc_true = d2r_dt2(t).squeeze() - gravity_acc
    q_acc_true = jnp.concatenate([jnp.array([0]), vec_acc_true], axis=-1)
    
    q_acc = quaternion_product(q_t_conjugate, quaternion_product(q_acc_true, q_t))
    _, acc = quaternion_scalar_and_vector(q_acc)
    
    return acc

# Initialize model
key = jax.random.PRNGKey(0)
pinn = PositionQuaternionNN(key)

# Vectorized versions of the models using vmap
vmap_pose = jax.vmap(pinn)
vmap_gyroscope_model = jax.vmap(gyroscope_model, in_axes=(None, 0))
vmap_accelerometer_model = jax.vmap(accelerometer_model, in_axes=(None, 0))

# Compute initial pinn prediction
ini_r_t_array, ini_q_t_array = vmap_pose(t_array)
print(f"Initial Positions: {ini_r_t_array}, Initial Quaternions: {ini_q_t_array}")

# Compute initial gyroscope readings for the array of time entries
ini_omega_t_array = vmap_gyroscope_model(pinn, t_array)
print(f"Initial Gyroscope readings: {ini_omega_t_array}")

# Compute initial accelerometer readings for the array of time entries
ini_acc_t_array = vmap_accelerometer_model(pinn, t_array)
print(f"Initial Accelerometer readings: {ini_acc_t_array}")

# Define loss function for pinn
def loss_fn(network, data_gyro, data_acc):
    
    pred_gyro = jax.vmap(gyroscope_model, in_axes=(None, 0))(network, t_array)
    pred_acc = jax.vmap(accelerometer_model, in_axes=(None, 0))(network, t_array)
    
    weight_gyro = 0.8
    weight_acc = 1.0 - weight_gyro
    
    loss_gyro = jnp.mean(jnp.square(pred_gyro - data_gyro))
    loss_acc = jnp.mean(jnp.square(pred_acc - data_acc))
    
    loss_bc_add = True
    
    if loss_bc_add:
        
        loss_bc_left_dot = jnp.dot(q_bc_left, network(t_array[0])[1])
        loss_bc_right_dot = jnp.dot(q_bc_right, network(t_array[-1])[1])
        
        loss_bc_left = 1.0 - jnp.abs(loss_bc_left_dot)
        loss_bc_right = 1.0 - jnp.abs(loss_bc_right_dot)
        loss_bc = 0.5 * loss_bc_left + 0.5 * loss_bc_right
        
        loss_total = 0.3 * loss_bc + 0.6 * loss_gyro + 0.1 * loss_acc
        
    else:
        
        loss_total = weight_gyro * loss_gyro + weight_acc * loss_acc
    
    return loss_total

# Compute initial loss
ini_loss_val = loss_fn(pinn, data_gyro, data_acc)
print(f"Initial Loss value: {ini_loss_val}")

# Hyperparameters
initial_learning_rate = 1e-3
decay_rate = 0.9
decay_steps = 500  # Define the number of steps after which the learning rate decays
n_epochs = 5000

# Define the learning rate schedule
schedule = optax.exponential_decay(
    init_value=initial_learning_rate,
    transition_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Training Loop
optimizer = optax.adam(schedule)
opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

@eqx.filter_jit
def make_step(network, optimizer_state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(network, data_gyro, data_acc)
    network_updates, new_optimizer_state = optimizer.update(grad, optimizer_state, network)
    new_network = eqx.apply_updates(network, network_updates)
    return new_network, new_optimizer_state, loss

start_time = time.time()
loss_history = []
for epoch in range(n_epochs):
    pinn, opt_state, loss = make_step(pinn, opt_state)
    loss_history.append(loss)
    print(f"Epoch: {epoch}, loss: {loss}")
end_time = time.time()

print("Execution time: %s seconds" % (end_time - start_time))

# Compute final pinn prediction
vmap_pose_trained = jax.vmap(pinn)
_, final_q_t_array = vmap_pose_trained(t_array)
print(f"Final Quaternions prediction: {final_q_t_array}")
    
# Compute final gyroscope predictions for the array of time entries
final_omega_t_array = vmap_gyroscope_model(pinn, t_array)
print(f"Final Gyroscope prediction: {final_omega_t_array}")

# Compute final accelerometer prediction for the array of time entries
final_acc_t_array = vmap_accelerometer_model(pinn, t_array)
print(f"Final Accelerometer prediction: {final_acc_t_array}")

loss_bc_left_dot = jnp.dot(q_bc_left, final_q_t_array[0])
loss_bc_right_dot = jnp.dot(q_bc_right, final_q_t_array[-1])
loss_bc_left_val = 1.0 - jnp.abs(loss_bc_left_dot)
loss_bc_right_val = 1.0 - jnp.abs(loss_bc_right_dot)
loss_bc_val = 0.5 * loss_bc_left_val + 0.5 * loss_bc_right_val
print(f"Boundary loss value: {loss_bc_val}")

# Extract inclination and heading from quaternions
def quat_to_rpy(quat):
    w, x, y, z = quat
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jnp.arcsin(2.0 * (w * y - z * x))
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.array([roll, pitch, yaw])

# Vectorized conversion
rpy_angles_ref_rad = jax.vmap(quat_to_rpy)(data_quat)
rpy_angles_pinn_rad = jax.vmap(quat_to_rpy)(final_q_t_array)

# Convert the angles from radians to degrees
rpy_angles_ref = jnp.degrees(rpy_angles_ref_rad)
rpy_angles_pinn = jnp.degrees(rpy_angles_pinn_rad)

# Plot loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss history")
plt.show()

# Plot loss history on log scale
plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.title("loss history on log scale")
plt.show()

# Plot final prediction for accelerometer
plt.figure()
for i in range(final_acc_t_array.shape[1]):
    if i == 0:
        plt.plot(t_array, data_acc[:, i], 'r--', label="imu data")
        plt.plot(t_array, final_acc_t_array[:, i], 'b', label="pinn prediction")
    else:
        plt.plot(t_array, data_acc[:, i], 'r--')
        plt.plot(t_array, final_acc_t_array[:, i], 'b')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r'a $(m/{s^2})$')
plt.title("accelerometer results")

# Plot final prediction for gyroscope
plt.figure()
for i in range(final_omega_t_array.shape[1]):
    if i == 0:
        plt.plot(t_array, data_gyro[:, i], 'r--', label="imu data")
        plt.plot(t_array, final_omega_t_array[:, i], 'b', label="pinn prediction")
    else:
        plt.plot(t_array, data_gyro[:, i], 'r--')
        plt.plot(t_array, final_omega_t_array[:, i], 'b')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r'$\omega$ (rad/s)')
plt.title("gyroscope results")

# Plot final prediction for orientation in terms of quaternions
quat_column_names = list(df_ref.columns[1:])
plt.figure()
[plt.plot(t_ref_array, data_quat[:, i], '--', label=quat_column_names[i]) for i in range(data_quat.shape[1])]
for i in range(final_q_t_array.shape[1]):
    if i == 0:
        plt.plot(t_array, final_q_t_array[:, i], 'b', label="pinn prediction")
    else:
        plt.plot(t_array, final_q_t_array[:, i], 'b')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("quaternion data")
plt.title("orientation results (quaternions)")

# Plot final prediction for roll
plt.figure()
plt.plot(t_ref_array, rpy_angles_ref[:,0], 'r--', label="reference data")
plt.plot(t_array, rpy_angles_pinn[:,0], 'b', label="pinn prediction")
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("roll (degrees)")
plt.title("roll results")

# Plot final prediction for pitch
plt.figure()
plt.plot(t_ref_array, rpy_angles_ref[:,1], 'r--', label="reference data")
plt.plot(t_array, rpy_angles_pinn[:,1], 'b', label="pinn prediction")
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("pitch (degrees)")
plt.title("pitch results")

# Plot final prediction for yaw
plt.figure()
plt.plot(t_ref_array, rpy_angles_ref[:,2], 'r--', label="reference data")
plt.plot(t_array, rpy_angles_pinn[:,2], 'b', label="pinn prediction")
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("yaw (degrees)")
plt.title("yaw results")