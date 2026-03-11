import jax.numpy as jnp
import numpy as np
import jax
import os
import pickle


def get_pathfinder_datasets(data_dir, seed=0, resolution=32, normalize=False, split='easy'):
    if split == "easy":
        diff_level = "curv_baseline"
    elif split == "intermediate":
        diff_level = "curv_contour_length_9"
    elif split == "hard":
        diff_level = "curv_contour_length_14"
    else:
        raise ValueError("split must be in ['easy', 'intermediate', 'hard'].")

    with open(os.path.join(data_dir, f"pathfinder{resolution}-{diff_level}.pickle"), "rb") as f:
        ds_list = pickle.load(f)

    rng = np.random.default_rng(seed)
    rng.shuffle(ds_list)

    bp80 = int(len(ds_list) * 0.8)
    bp90 = int(len(ds_list) * 0.9)
    train_set = ds_list[:bp80]
    val_set = ds_list[bp80:bp90]
    test_set = ds_list[bp90:]

    def convert_pickle_to_numpy_tensor(pickle_set):
        output = {"input_ids_0": [], "label": []}
        for entry in range(len(pickle_set)):
            input_ids = np.array(pickle_set[entry]["input_ids_0"], dtype=np.float32)
            input_ids = input_ids / 255 if normalize else input_ids
            label = np.array(pickle_set[entry]["label"], dtype=np.int32)
            output["input_ids_0"].append(input_ids)  # Append tensors
            output["label"].append(label)
        return output

    train_out = convert_pickle_to_numpy_tensor(train_set)
    val_out = convert_pickle_to_numpy_tensor(val_set)
    test_out = convert_pickle_to_numpy_tensor(test_set)

    train_images, train_labels = np.stack(train_out["input_ids_0"])[..., None], np.stack(train_out["label"])
    val_images, val_labels = np.stack(val_out["input_ids_0"])[..., None], np.stack(val_out["label"])
    test_images, test_labels = np.stack(test_out["input_ids_0"])[..., None], np.stack(test_out["label"])

    train_images, train_labels_one_hot = jnp.array(train_images), jnp.array(jax.nn.one_hot(train_labels, num_classes=2))
    val_images, val_labels_one_hot = jnp.array(val_images), jnp.array(jax.nn.one_hot(val_labels, num_classes=2))
    test_images, test_labels_one_hot = jnp.array(test_images), jnp.array(jax.nn.one_hot(test_labels, num_classes=2))

    return (jax.device_put(train_images), jax.device_put(train_labels_one_hot),
            jax.device_put(val_images), jax.device_put(val_labels_one_hot),
            jax.device_put(test_images), jax.device_put(test_labels_one_hot))
