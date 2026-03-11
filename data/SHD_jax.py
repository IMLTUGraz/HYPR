import h5py
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import tonic
from tonic import transforms
import os
import sys

def load_shd_file(hdf5_filepath: str):
    print(f"Loading SHD file from {hdf5_filepath}")
    with h5py.File(hdf5_filepath, 'r') as f:
        spike_times = f['spikes/times']
        spike_units = f['spikes/units']
        labels = f['labels'][:]
        times_list = [np.array(spike_times[i]) for i in range(len(spike_times))]
        units_list = [np.array(spike_units[i]) for i in range(len(spike_units))]
    return times_list, units_list, labels

def bin_spike_train(times, units, bin_size_ms: float, num_channels: int, max_time_ms: float):
    T = int(np.ceil(max_time_ms / bin_size_ms))
    raster = np.zeros((T, num_channels), dtype=np.float32)
    bin_indices = (times / (bin_size_ms / 1000.0)).astype(np.int32)
    valid = bin_indices < T
    bin_indices = bin_indices[valid]
    # reduce to num_channels
    # valid_units = (units[valid]  * num_channels) // 700
    valid_units = (units[valid].astype(float) * num_channels) / 700.0
    valid_units = valid_units.astype(np.int16)
    np.add.at(raster, (bin_indices, valid_units), 1)
    return raster

# def prepare_shd_dataset(hdf5_filepath: str, bin_size_ms: float, num_channels: int, max_time_ms: float):
#     times_list, units_list, labels = load_shd_file(hdf5_filepath)
#     binned_samples = [
#         bin_spike_train(times, units, bin_size_ms, num_channels, max_time_ms)
#         for times, units in zip(times_list, units_list)
#     ]


#     # X = np.stack(binned_samples, axis=0)
#     y = get_one_hot(np.array(labels), 20)
#     # X_jax = jnp.array(X)
#     # y_jax = jnp.array(y)
#     return X_jax, y_jax

def prepare_shd_dataset(hdf5_filepath: str, bin_size_ms: float, num_channels: int, max_time_ms: float):
    times_list, units_list, labels = load_shd_file(hdf5_filepath)
    binned_samples = [
        bin_spike_train(times, units, bin_size_ms, num_channels, max_time_ms)
        for times, units in zip(times_list, units_list)
    ]
    X = np.stack(binned_samples, axis=0)
    y = get_one_hot(np.array(labels), 20)

    tiny_shd_x = X[np.argmax(y, axis=-1)<3]
    tiny_shd_y = y[np.argmax(y, axis=-1)<3]
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)
    return X_jax, y_jax

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes]).astype(np.float32)



# # mannual dataloader 
# class SHDDataLoader:
#     def __init__(self, X, y, batch_size, shuffle=True, seed=0):
#         self.X = X
#         self.y = y
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.seed = seed
#         self.num_samples = X.shape[0]

#     def __iter__(self):
#         # Create permutation
#         if self.shuffle:
#             rng = np.random.default_rng(self.seed)
#             indices = rng.permutation(self.num_samples)
#         else:
#             indices = np.arange(self.num_samples)

#         # Generate batches
#         for start_idx in range(0, self.num_samples, self.batch_size):
#             batch_indices = indices[start_idx:start_idx + self.batch_size]
#             yield self.X[batch_indices], self.y[batch_indices]

# # test if it works
# # Load dataset
# X_jax, y_jax = prepare_shd_dataset(
#     hdf5_filepath="shd_train.h5",
#     bin_size_ms=1.0,
#     num_channels=700,
#     max_time_ms=1400.0
# )

# # Create DataLoader
# train_loader = SHDDataLoader(X_jax, y_jax, batch_size=32, shuffle=True, seed=42)

# # Iterate over one epoch
# for batch_X, batch_y in train_loader:
#     print("Batch X shape:", batch_X.shape)  # (32, T, 700)
#     print("Batch y shape:", batch_y.shape)  # (32,)
#     break



def convert_dataset_to_jax(dataset, sensor_size, time_window):
    spike_tensors, labels = [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        
        # print(f"Sample {i}: {sample}")  
        
        spikes = sample[0]  
        label = sample[1]   
        
        spikes_dense = spikes.to_dense()

        spike_tensors.append(spikes_dense)
        labels.append(label)
    
    jax_spikes = jnp.array([spike.cpu().numpy() for spike in spike_tensors])  
    jax_labels = jnp.array(labels)
    
    return jax_spikes, jax_labels

def get_shd_dataloaders( time_window):
    data_path = "./data/SHD" 
    sensor_size = tonic.datasets.SHD.sensor_size  
    
    os.makedirs(data_path, exist_ok=True)
    
    transform = transforms.ToSparseTensor(sensor_size=sensor_size, time_window=time_window)
    
    train_dataset = tonic.datasets.SHD(save_to=data_path, train=True, transform=transform)
    test_dataset = tonic.datasets.SHD(save_to=data_path, train=False, transform=transform)
    
    jax_train_spikes, jax_train_labels = convert_dataset_to_jax(train_dataset, sensor_size, time_window)
    jax_test_spikes, jax_test_labels = convert_dataset_to_jax(test_dataset, sensor_size, time_window)
    
    return (jax_train_spikes, jax_train_labels), (jax_test_spikes, jax_test_labels), sensor_size

# time_window = 100  
# (jax_train_spikes, jax_train_labels), (jax_test_spikes, jax_test_labels), sensor_size = get_shd_dataloaders( time_window)


# print("Train spikes shape:", jax_train_spikes.shape)
# print("Train labels shape:", jax_train_labels.shape)











