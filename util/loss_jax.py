import jax.numpy as jnp
from optax import losses

def masked_cross_entropy_loss(logits, labels, ignore_mask, batch_axis=0, reduce_over_batch="mean"):
    loss = losses.safe_softmax_cross_entropy(logits=logits, labels=labels) * ignore_mask
    if reduce_over_batch == "mean":
        loss = jnp.mean(loss, axis=batch_axis, keepdims=True)
    elif reduce_over_batch == "sum":
        loss = jnp.sum(loss, axis=batch_axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reduce_over_batch: {reduce_over_batch}, must be 'mean' or 'sum'")
    return jnp.squeeze(loss, axis=batch_axis)


def cross_entropy_loss(logits, labels, batch_axis=0, reduce_over_batch="mean"):
    loss = losses.safe_softmax_cross_entropy(logits=logits, labels=labels)
    if reduce_over_batch == "mean":
        loss = jnp.mean(loss, axis=batch_axis, keepdims=True)
    elif reduce_over_batch == "sum":
        loss = jnp.sum(loss, axis=batch_axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reduce_over_batch: {reduce_over_batch}, must be 'mean' or 'sum'")
    return jnp.squeeze(loss, axis=batch_axis)


def masked_seq_cross_entropy_loss(logits, labels, ignore_mask, batch_axis=0, reduce_over_batch="mean",  seq_axis=1, reduce_over_sequence="sum"):
    """Cross Entropy Loss

    Args:
        logits (jnp.Array): unnormalized logits of (batch_size x seq_len x num_classes)
        labels (jnp.Array): labels of shape (batch_size x 1 x num_classes)
        batch_axis (int, optional): Batch dim. Defaults to 0.
        reduce_over_batch (str, optional): reduction over batch axis, 'mean' or 'sum'. Defaults to "mean".
        seq_axis (int, optional): Sequence axis. Defaults to 1.
        reduce_over_sequence (str, optional): reduction over sequence axis, 'mean' or 'sum'. Defaults to "sum".

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        jnp.Array: scalar loss
    """
    loss = losses.safe_softmax_cross_entropy(logits=logits, labels=labels) * ignore_mask

    if reduce_over_batch == "mean":
        loss = jnp.mean(loss, axis=batch_axis, keepdims=True)
    elif reduce_over_batch == "sum":
        loss = jnp.sum(loss, axis=batch_axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reduce_over_batch: {reduce_over_batch}, must be 'mean' or 'sum'")
    
    if reduce_over_sequence == "mean":
        loss = jnp.mean(loss, axis=seq_axis, keepdims=True)
    elif reduce_over_sequence == "sum":
        loss = jnp.sum(loss, axis=seq_axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reduce_over_sequence: {reduce_over_sequence}, must be 'mean' or 'sum'")
    
    return jnp.squeeze(loss, axis=(batch_axis, seq_axis))
