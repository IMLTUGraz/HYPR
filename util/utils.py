import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax
from flax import nnx


def sample_from_gaussian(mean, stddev, shape, rng_key):
    """Samples from a Gaussian distribution with given mean and stddev."""
    return jax.random.normal(rng_key, shape=shape) * stddev + mean

def sample_from_range(range, shape, rng_key):
    return jax.random.uniform(rng_key, shape=shape, minval=range[0], maxval=range[1])


def masked_sum_of_softmaxes(logits, ignore_mask):
    """Computs the sum of softmaxes. Assumes (batch_size x seq_len x num_classes)
    """
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * ignore_mask[..., None], axis=1)

def custom_xavier_uniform_initializer(fan_in, fan_out):
    def init(key, shape, dtype=jnp.float32):
        # Xavier uniform initialization
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
    return init

def custom_zeros_initializer():
    def init(key, shape, dtype=jnp.float32):
        # Xavier uniform initialization
        return jnp.zeros(shape, dtype=dtype)
    return init

def get_initializer(initializer, fan_in, fan_out):
    if initializer == "xavier_uniform_custom":
        return custom_xavier_uniform_initializer(fan_in, fan_out)
    elif initializer == "xavier_uniform":
        return nnx.initializers.xavier_uniform()
    elif initializer == "xavier_normal":
        return nnx.initializers.xavier_normal()
    elif initializer == "lecun_uniform":
        return nnx.initializers.lecun_uniform()
    elif initializer == "orthogonal":
        return nnx.initializers.orthogonal()
    elif initializer == "zeros":
        return custom_zeros_initializer()
    else:
        raise ValueError(f"Unknown initializer: {initializer}")


def compute_grad_norm(grads):
    # Flatten the gradients into a single 1D array
    flat_grads, _ = ravel_pytree(grads)
    # Compute the L2 norm of the flattened gradients
    return jnp.linalg.norm(flat_grads)

def track_detailed_grad_norms(run, grads, step):
    grad_norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    grad_norms_flat = jax.tree_util.tree_flatten_with_path(grad_norms)[0]
    grad_norms_dict = {jax.tree_util.keystr(k): v for k, v in grad_norms_flat}
    run.track(grad_norms_dict, step=step)

def maybe_clip_gradient_norm(grads, grad_clip_val=None):
    if grad_clip_val is not None:
        grad_norm = compute_grad_norm(grads)
        divisor = jnp.maximum(1.0, grad_norm / grad_clip_val)
        grads = jax.tree_util.tree_map(lambda x: x / divisor, grads)
    return grads

def step_lr_schedule(init_lr, step_size, gamma):
    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)  # make sure it's a float32 tensor
        factor = gamma ** (jnp.floor(step / step_size))
        return init_lr * factor
    return schedule



