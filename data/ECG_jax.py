import os
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import scipy.io



def convert_dataset_wtime(mat_data):
    X = mat_data["x"]
    Y = np.argmax(mat_data["y"], axis=-1)  # one-hot to int
    t = mat_data["t"]
    d1, d2 = t.shape
    dt = np.zeros((d1, d2))
    for trace in range(d1):
        dt[trace, 0] = 1
        dt[trace, 1:] = t[trace, 1:] - t[trace, :-1]
    return dt, X, Y

def load_ecg_dataset_jax(data_dir, seed):
    data_dir = Path(data_dir)

    files = {
        "train": ("ECG/QTDB_train.mat", "https://raw.githubusercontent.com/byin-cwi/Efficient-spiking-networks/main/data/QTDB_train.mat"),
        "test": ("ECG/QTDB_test.mat", "https://raw.githubusercontent.com/byin-cwi/Efficient-spiking-networks/main/data/QTDB_test.mat"),
    }

    # for name, (rel_path, url) in files.items():
    #     download_file(url, data_dir / rel_path)

    train_mat = scipy.io.loadmat(data_dir / files["train"][0])
    test_mat = scipy.io.loadmat(data_dir / files["test"][0])

    train_dt, train_x, train_y = convert_dataset_wtime(train_mat)

    # Split seeded randomized validation set
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_x))
    train_x_shuffled, train_y_shuffled = train_x[indices], train_y[indices]
    train_size = int(len(train_x_shuffled) * 0.9)
    train_x, train_y = train_x_shuffled[:train_size], train_y_shuffled[:train_size]
    val_x, val_y = train_x_shuffled[train_size:], train_y_shuffled[train_size:]

    test_dt, test_x, test_y = convert_dataset_wtime(test_mat)

    train_x, train_y, train_dt = jnp.array(train_x), jnp.array(jax.nn.one_hot(train_y , 6)), jnp.array(train_dt)
    test_x, test_y, test_dt = jnp.array(test_x),  jnp.array(jax.nn.one_hot(test_y , 6)), jnp.array(test_dt)
    val_x, val_y = jnp.array(val_x), jnp.array(jax.nn.one_hot(val_y , 6))

    return  train_x, train_y, val_x, val_y, test_x, test_y
    
