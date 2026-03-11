import os
import numpy as np
import jax_dataloader as jdl
import jax
import jax.numpy as jnp
from flax import nnx
from data.SHD_jax import prepare_shd_dataset
from data.toy_task_cue_generation import cue_data_jax
from data.ECG_jax import load_ecg_dataset_jax
from data.SMNIST_jax import load_sequential_mnist
from data.SCIFAR_jax import load_sequential_cifar
from data.PATHFINDER_jax import get_pathfinder_datasets


def load_dataset(data_dir, dataset_cfg, key):
    if dataset_cfg.dataset_name == "cue":
        train_input, train_target, test_input, test_target = cue_data_jax(
            key,
            num_sample=dataset_cfg.num_samples,
            sequence_length=dataset_cfg.seq_len,
            input_size=dataset_cfg.input_dim,
            prob_A=dataset_cfg.prob_A,
            prob_B=dataset_cfg.prob_B,
            prob_cue=dataset_cfg.prob_cue,
            delay_length=dataset_cfg.delay,
            train_ratio=dataset_cfg.train_ratio,
        )
        # size of train_input in MB:
        print(f"train_input size: {train_input.nbytes / 1024 ** 2:.2f} MB")
        train_ds = jdl.ArrayDataset(jax.device_put(train_input), jax.device_put(train_target), asnumpy=False)
        train_dl = jdl.DataLoader(
            train_ds,
            "jax",
            batch_size=dataset_cfg.batch_size,
            drop_last=True,
            shuffle=True,
        )
        test_ds = jdl.ArrayDataset(jax.device_put(test_input), jax.device_put(test_target), asnumpy=False)
        test_dl = jdl.DataLoader(
            test_ds,
            "jax",
            batch_size=dataset_cfg.batch_size,
            drop_last=False,
            shuffle=False,
        )
        input_dim = dataset_cfg.input_dim
        total_seq_len = dataset_cfg.total_seq_len
        batch_size = dataset_cfg.batch_size
        num_classes = 2
        return (
            test_dl,
            test_dl,
            train_dl,
            input_dim,
            total_seq_len,
            batch_size,
            num_classes,
        )

    elif dataset_cfg.dataset_name == "shd":
        train_file = os.path.join(
            data_dir, dataset_cfg.subdirectory, dataset_cfg.train_file_name
        )
        test_file = os.path.join(
            data_dir, dataset_cfg.subdirectory, dataset_cfg.test_file_name
        )
        bin_size = dataset_cfg.bin_size_ms
        num_channels = dataset_cfg.num_channels
        max_time = dataset_cfg.max_time_ms
        batch_size = dataset_cfg.batch_size

        X_train, y_train = prepare_shd_dataset(
            train_file, bin_size, num_channels, max_time
        )

        # Split seeded randomized validation set
        rng = np.random.default_rng(dataset_cfg.seed)
        indices = rng.permutation(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        train_size = int(len(X_train_shuffled) * 0.9)
        X_train, y_train = X_train_shuffled[:train_size], y_train_shuffled[:train_size]
        X_val, y_val = X_train_shuffled[train_size:], y_train_shuffled[train_size:]

        X_test, y_test = prepare_shd_dataset(
            test_file, bin_size, num_channels, max_time
        )

        if dataset_cfg.clip_to_1:
            X_train = jnp.clip(X_train, 0, 1)
            X_val = jnp.clip(X_val, 0, 1)
            X_test = jnp.clip(X_test, 0, 1)
        # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        # (jax_train_spikes, jax_train_labels), (jax_test_spikes, jax_test_labels), sensor_size = get_shd_dataloaders( bin_size)

        # X_train = jax.random.normal(key, (8156, 250, 700))
        # y_train = jax.random.normal(key, (8156, 20))
        input_dim = X_train.shape[-1]
        total_seq_len = X_train.shape[-2]
        train_ds = jdl.ArrayDataset(jax.device_put(X_train), jax.device_put(y_train), asnumpy=False)
        train_dl = jdl.DataLoader(
            train_ds,
            "jax",
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        val_ds = jdl.ArrayDataset(jax.device_put(X_val), jax.device_put(y_val), asnumpy=False)
        val_dl = jdl.DataLoader(
            val_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )

        test_ds = jdl.ArrayDataset(jax.device_put(X_test), jax.device_put(y_test), asnumpy=False)
        test_dl = jdl.DataLoader(
            test_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )

        return (
            test_dl,
            val_dl,
            train_dl,
            input_dim,
            total_seq_len,
            batch_size,
            dataset_cfg.num_classes,
        )

    elif dataset_cfg.dataset_name == "ecg":
        file = os.path.join(
            data_dir, dataset_cfg.subdirectory)

        batch_size = dataset_cfg.batch_size
        seed = dataset_cfg.seed

        X_train, y_train, X_val, y_val, X_test, y_test = load_ecg_dataset_jax(data_dir, seed)

        input_dim = X_train.shape[-1]
        total_seq_len = X_train.shape[-2]
        train_ds = jdl.ArrayDataset(jax.device_put(X_train), jax.device_put(y_train), asnumpy=False)
        train_dl = jdl.DataLoader(
            train_ds,
            "jax",
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        val_ds = jdl.ArrayDataset(jax.device_put(X_val), jax.device_put(y_val), asnumpy=False)
        val_dl = jdl.DataLoader(
            val_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )

        test_ds = jdl.ArrayDataset(jax.device_put(X_test), jax.device_put(y_test), asnumpy=False)
        test_dl = jdl.DataLoader(
            test_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )

        return (
            test_dl,
            val_dl,
            train_dl,
            input_dim,
            total_seq_len,
            batch_size,
            dataset_cfg.num_classes,
        )

    elif dataset_cfg.dataset_name == "smnist" or dataset_cfg.dataset_name == "scifar":
        file = os.path.join(data_dir, dataset_cfg.subdirectory)
        seed = dataset_cfg.seed
        permute = dataset_cfg.permute
        if dataset_cfg.permute:
            print("Permutation is ON")

        if dataset_cfg.dataset_name == "smnist":
            X_train, y_train, X_val, y_val, X_test, y_test = load_sequential_mnist(file, seed, permute)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = load_sequential_cifar(file, seed, permute, dataset_cfg.color)

        input_dim = X_train.shape[-1]
        total_seq_len = X_train.shape[-2]
        batch_size = dataset_cfg.batch_size

        train_ds = jdl.ArrayDataset(X_train, y_train, asnumpy=False)
        train_dl = jdl.DataLoader(
            train_ds,
            "jax",
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        val_ds = jdl.ArrayDataset(X_val, y_val, asnumpy=False)
        val_dl = jdl.DataLoader(
            val_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )

        test_ds = jdl.ArrayDataset(X_test, y_test, asnumpy=False)
        test_dl = jdl.DataLoader(
            test_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )

        num_classes = 10

        return (
            test_dl,
            val_dl,
            train_dl,
            input_dim,
            total_seq_len,
            batch_size,
            num_classes,
        )

    elif dataset_cfg.dataset_name == "pathfinder":
        file = os.path.join(data_dir, dataset_cfg.subdirectory)
        seed = dataset_cfg.seed
        resolution = dataset_cfg.resolution
        normalize = dataset_cfg.normalize
        split = dataset_cfg.split

        X_train, y_train, X_val, y_val, X_test, y_test = get_pathfinder_datasets(file, seed, resolution, normalize, split)

        input_dim = X_train.shape[-1]
        total_seq_len = X_train.shape[-2]
        batch_size = dataset_cfg.batch_size

        train_ds = jdl.ArrayDataset(X_train, y_train, asnumpy=False)
        train_dl = jdl.DataLoader(
            train_ds,
            "jax",
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        val_ds = jdl.ArrayDataset(X_val, y_val, asnumpy=False)
        val_dl = jdl.DataLoader(
            val_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )

        test_ds = jdl.ArrayDataset(X_test, y_test, asnumpy=False)
        test_dl = jdl.DataLoader(
            test_ds,
            "jax",
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )

        return (
            test_dl,
            val_dl,
            train_dl,
            input_dim,
            total_seq_len,
            batch_size,
            dataset_cfg.num_classes,
        )

    else:
        print('Dataset name is invalid.')
