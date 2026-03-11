import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
import numpy as np

def get_one_hot(targets, nb_classes):
    res = jnp.eye(nb_classes)[jnp.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def cue_data_jax(
    key,
    num_sample,
    sequence_length,
    input_size,
    prob_A,
    prob_B,
    prob_cue,
    delay_length,
    train_ratio
):
    """
    Generates a balanced dataset with train-test split in JAX.

    Args:
        key: JAX PRNGKey for random generation.
        (Other args are the same as in the PyTorch version.)

    Returns:
        (train_input, train_target), (test_input, test_target)
    """

    def generate_sample(key, i):
        key, key_class, key_spike, key_cue = random.split(key, 4)

        target = random.randint(key_class, shape=(), minval=0, maxval=2)

        sample = jnp.zeros((sequence_length, input_size))

        def create_class_A(_):
            spikes = random.bernoulli(key_spike, prob_A, (sequence_length, 5))
            return sample.at[:,  :5].set(spikes)

        def create_class_B(_):
            spikes = random.bernoulli(key_spike, prob_B, (sequence_length, 5))
            return sample.at[:,  5:10].set(spikes)

        sample = jax.lax.cond(target == 0, create_class_A, create_class_B, operand=None)

        delay = jnp.zeros((delay_length, input_size))

        cue_spikes = random.bernoulli(key_cue, prob_cue, (sequence_length, 5))
        sample_cue = jnp.zeros((sequence_length, input_size))
        sample_cue = sample_cue.at[:,  10:15].set(cue_spikes)

        sample_full = jnp.concatenate([sample, delay, sample_cue], axis=0)
    
        return sample_full, get_one_hot(target, 2)




    # input_tensor, target_tensor, _ = jax.lax.fori_loop(
    #     0, num_sample, generate_sample, (input_tensor, target_tensor, key)
    # )
    keys = random.split(key, num_sample)
    input_tensor, target_tensor = jax.vmap(generate_sample, in_axes=(0, 0))(keys, jnp.arange(num_sample))

    # Train-test split
    key, subkey = random.split(key)
    indices = random.permutation(subkey, num_sample)
    num_train = int(num_sample * train_ratio)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_input = input_tensor[train_indices]
    train_target = target_tensor[train_indices]
    test_input = input_tensor[test_indices]
    test_target = target_tensor[test_indices]

    return train_input, train_target, test_input, test_target


## working on vectorized version

def cue_data_jax_vmap(
    key,
    num_sample,
    sequence_length,
    batch_size,
    input_size,
    prob_A,
    prob_B,
    prob_cue,
    delay_length,
    train_ratio
):
    
    total_length = sequence_length * 2 + delay_length  # sequence + delay + cue

    key, sample_key = random.split(key)
    sample_keys = random.split(sample_key, num_sample)

    def generate_single_sample(key):
        key_class, key_spike, key_cue = random.split(key, 3)

        n = random.randint(key_class, shape=(), minval=0, maxval=2)

        sample = jnp.zeros((sequence_length, batch_size, input_size))

        def create_class_A():
            spikes = random.bernoulli(key_spike, prob_A, (sequence_length, batch_size, 5))
            return sample.at[:, :, :5].set(spikes)

        def create_class_B():
            spikes = random.bernoulli(key_spike, prob_B, (sequence_length, batch_size, 5))
            return sample.at[:, :, 5:10].set(spikes)

        sample = jax.lax.cond(n == 0, create_class_A, create_class_B)

        delay = jnp.zeros((delay_length, batch_size, input_size))

        cue_spikes = random.bernoulli(key_cue, prob_cue, (sequence_length, batch_size, 5))
        sample_cue = jnp.zeros((sequence_length, batch_size, input_size))
        sample_cue = sample_cue.at[:, :, 10:15].set(cue_spikes)

        sample_full = jnp.concatenate([sample, delay, sample_cue], axis=0)
        return sample_full, n

    # Vectorize 
    samples, targets = vmap(generate_single_sample)(sample_keys)

    key, subkey = random.split(key)
    indices = random.permutation(subkey, num_sample)
    num_train = int(num_sample * train_ratio)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_input = samples[train_indices]
    train_target = targets[train_indices]
    test_input = samples[test_indices]
    test_target = targets[test_indices]

    return train_input, train_target, test_input, test_target










# key = random.PRNGKey(0)
# train_input, train_target, test_input, test_target = cue_data_jax_vmap(
#     key,
#     num_sample=100,
#     sequence_length=20,
#     batch_size=1,
#     input_size=15,
#     prob_A=0.2,
#     prob_B=0.3,
#     prob_cue=0.5,
#     delay_length=20,
#     train_ratio=0.8
# )


# def plot_spike_sample(sample, target, title=None):
    
#     time_steps, batch_size, input_size = sample.shape
#     sample = np.array(sample[:, 0, :])  

#     fig, ax = plt.subplots(figsize=(10, 4))

#     for neuron in range(input_size):
#         spike_times = np.where(sample[:, neuron] > 0)[0]
#         ax.vlines(spike_times, neuron + 0.5, neuron + 1.5)

#     ax.set_ylim(0.5, input_size + 0.5)
#     ax.set_xlim(0, time_steps)
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Neuron Index")
#     if title:
#         ax.set_title(f"{title} (Target: {target})")
#     else:
#         ax.set_title(f"Spike Sample (Target: {target})")
#     plt.tight_layout()
#     plt.show()

# def sample_and_plot(train_input, train_target, num_samples=30):
    
#     indices = np.random.choice(train_input.shape[0], size=num_samples, replace=False)
#     for idx in indices:
#         plot_spike_sample(train_input[idx], train_target[idx])

# #Example usage after generating data:
# sample_and_plot(train_input, train_target, num_samples=1)
