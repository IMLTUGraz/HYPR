import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax


def load_sequential_cifar(data_dir, seed=0, permute=False, color=False):
    ds_train = tfds.load("cifar10", split="train", data_dir=data_dir, as_supervised=True, batch_size=-1)
    ds_test = tfds.load("cifar10", split="test", data_dir=data_dir, as_supervised=True, batch_size=-1)

    train_images, train_labels = tfds.as_numpy(ds_train)
    test_images, test_labels = tfds.as_numpy(ds_test)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(1024) if permute else np.arange(1024)
    indices = rng.permutation(len(train_images))

    if color:
        train_images = train_images.reshape(-1, 1024, 3).astype(np.float32) / 255.0
        test_images = test_images.reshape(-1, 1024, 3).astype(np.float32) / 255.0

        train_images = train_images[:, perm, :]
        test_images = test_images[:, perm, :]
    else:
        train_images = tf.image.rgb_to_grayscale(train_images)
        test_images = tf.image.rgb_to_grayscale(test_images)

        train_images = train_images.numpy().reshape(-1, 1024).astype(np.float32) / 255.0
        test_images = test_images.numpy().reshape(-1, 1024).astype(np.float32) / 255.0

        train_images = train_images[:, perm]
        test_images = test_images[:, perm]

        # add a channel dimension
        train_images = train_images[..., None]
        test_images = test_images[..., None]

    train_images_shuffled, train_labels_shuffled = train_images[indices], train_labels[indices]
    train_size = int(len(train_images_shuffled) * 0.9)
    train_images, train_labels = train_images_shuffled[:train_size], train_labels_shuffled[:train_size]
    val_images, val_labels = train_images_shuffled[train_size:], train_labels_shuffled[train_size:]

    # convert labels to one-hot encoding
    train_labels_one_hot = jax.nn.one_hot(train_labels, num_classes=10)
    val_labels_one_hot = jax.nn.one_hot(val_labels, num_classes=10)
    test_labels_one_hot = jax.nn.one_hot(test_labels, num_classes=10)

    return (jax.device_put(train_images), jax.device_put(train_labels_one_hot),
            jax.device_put(val_images), jax.device_put(val_labels_one_hot),
            jax.device_put(test_images), jax.device_put(test_labels_one_hot))
