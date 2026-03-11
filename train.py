import jax.numpy as jnp
from flax import nnx
import jax
from util.loss_jax import masked_seq_cross_entropy_loss, cross_entropy_loss
from util.utils import masked_sum_of_softmaxes, maybe_clip_gradient_norm


@nnx.scan(in_axes=(None, None, 1, 1, nnx.Carry), out_axes=(0, nnx.Carry))
def subseq_eval(model, targets_expanded, inputs, ignore_mask, carry):
    cum_prediction, prev_state = carry
    loss, (output, state) = run_model_hypr(
        model, inputs, ignore_mask, targets_expanded, prev_state
    )
    cum_prediction += jnp.sum(output * ignore_mask[..., None], axis=1)
    return loss, (cum_prediction, state)


@nnx.jit()
def eval_model_online(
    model, inputs, ignore_mask, target, init_online_eval, state, test_metrics
):
    if target.ndim == 2:
        targets_expanded = target[:, None, :]
    else:
        targets_expanded = target
    loss, (cum_prediction, _) = subseq_eval(
        model, targets_expanded, inputs, ignore_mask, (init_online_eval, state)
    )

    timesteps_with_nonzero_loss = jnp.sum(
        ignore_mask.reshape((ignore_mask.shape[0], -1))[0]
    )
    loss = loss / timesteps_with_nonzero_loss

    test_metrics.update(
        logits=cum_prediction,
        labels=jnp.argmax(targets_expanded[:, -1], axis=-1).astype(jnp.int32),
        values=loss,
    )
    return None


@nnx.jit(static_argnums=(6, 7))
def eval_model(
    model,
    inputs,
    ignore_mask,
    target,
    state,
    test_metrics,
    loss_aggregation,
    prediction_mode,
):
    ((output, spike_rates), state) = model(inputs, state, return_spike_rates=True)

    loss = aggregate_loss(
        output, target, ignore_mask, loss_aggregation
    )
    logits = get_prediction(output, ignore_mask, prediction_mode)

    labels = jnp.argmax(target, axis=-1).astype(jnp.int32)
    test_metrics.update(logits=logits, labels=labels, values=loss)

    return spike_rates


def run_model_hypr(model, inputs, ignore_mask, target, state):
    (output, state) = model(inputs, state, return_spike_rates=False)

    # output shape: (batch_size x seq_len x num_classes)
    # target shape: (batch_size x 1 x num_classes)
    loss = masked_seq_cross_entropy_loss(output, target, ignore_mask)
    return loss, (output, state)


@nnx.scan(in_axes=(None, None, 1, 1, nnx.Carry), out_axes=(0, nnx.Carry))
# @nnx.jit
def scan_hypr_subseq(model, targets_expanded, inputs, ignore_mask, carry):
    grads_sum, prev_state = carry
    grad_fn = nnx.value_and_grad(run_model_hypr, has_aux=True)
    (loss, (output, state)), grads = grad_fn(
        model, inputs, ignore_mask, targets_expanded, prev_state
    )
    grads_sum = jax.tree_util.tree_map(lambda x, y: x + y, grads_sum, grads)
    return (loss, output), (grads_sum, state)


@nnx.jit(static_argnums=(9))
def train_batch_hypr(
    model,
    data_split,
    targets,
    init_cell_state,
    grads_sum_init,
    optimizer,
    train_metrics,
    grad_clip_val,
    ignore_mask,
    prediction_mode,
):
    if targets.ndim == 2:
        targets_expanded = targets[:, None, :]
    else:
        targets_expanded = targets

    (loss, output), (grads_sum, _) = scan_hypr_subseq(
        model,
        targets_expanded,
        data_split,
        ignore_mask,
        (grads_sum_init, init_cell_state),
    )
    # sum loss over subsequences
    loss = jnp.sum(loss)

    # divide by the number of timesteps where we have a loss.
    timesteps_with_nonzero_loss = jnp.sum(
        ignore_mask.reshape((ignore_mask.shape[0], -1))[0]
    )
    grads_mean = jax.tree_util.tree_map(lambda x: x / timesteps_with_nonzero_loss, grads_sum)
    loss = loss / timesteps_with_nonzero_loss

    grads_mean_clipped = maybe_clip_gradient_norm(grads_mean, grad_clip_val)
    optimizer.update(grads_mean_clipped)
    model.apply_parameter_constraints()

    output = output.swapaxes(0, 1)

    # bring output and ignore_mask back to the original shape (full sequence)
    # This is needed for SHD, but for other datasets, the prediction might work differently (online).
    # Note, that the sum-of-softmaxes is NOT online!!!
    outputs_reshaped = output.reshape(
        (output.shape[0], output.shape[1] * output.shape[2], -1)
    )
    ignore_mask_reshaped = ignore_mask.reshape(
        (ignore_mask.shape[0], ignore_mask.shape[1] * ignore_mask.shape[2])
    )
    logits = get_prediction(outputs_reshaped, ignore_mask_reshaped, prediction_mode)
    train_metrics.update(
        logits=logits,
        labels=jnp.argmax(targets, axis=-1).astype(jnp.int32),
        values=loss,
    )
    return loss, grads_mean


def run_model_bptt(
    model, inputs, ignore_mask, target, init_state, loss_aggregation, prediction_mode
):
    output, _ = model(inputs, init_state)
    loss = aggregate_loss(output, target, ignore_mask, loss_aggregation)
    logits = get_prediction(output, ignore_mask, prediction_mode)
    return loss, (output, logits)


@nnx.jit(static_argnums=(8, 9))
def train_batch_bptt(
    model,
    data_split,
    target,
    init_state,
    optimizer,
    train_metrics,
    grad_clip_val,
    ignore_mask,
    loss_aggregation,
    prediction_mode,
):
    grad_fn = nnx.value_and_grad(run_model_bptt, has_aux=True)
    (loss, (output, logits)), grads = grad_fn(
        model,
        data_split,
        ignore_mask,
        target,
        init_state,
        loss_aggregation,
        prediction_mode,
    )

    grads_clipped = maybe_clip_gradient_norm(grads, grad_clip_val)
    optimizer.update(grads_clipped)
    model.apply_parameter_constraints()
    labels = jnp.argmax(target, axis=-1).astype(jnp.int32)

    train_metrics.update(logits=logits, labels=labels, values=loss)
    return loss, grads


def aggregate_loss(output, target, ignore_mask, loss_aggregation):
    if loss_aggregation == "sum_of_softmax":
        if target.ndim == 3:
            raise ValueError("Target must be 2D for sum_of_softmax loss aggregation")
        logits = masked_sum_of_softmaxes(output, ignore_mask)
        # logits are masked
        loss = cross_entropy_loss(logits, target)

    elif loss_aggregation == "per_timestep_cross_entropy":
        if target.ndim == 2:
            target = target[:, None, :]
        loss = masked_seq_cross_entropy_loss(output, target, ignore_mask)
        timesteps_with_nonzero_loss = jnp.sum(
            ignore_mask.reshape((ignore_mask.shape[0], -1))[0]
        )
        loss = loss / timesteps_with_nonzero_loss

    elif loss_aggregation == "last_timestep_cross_entropy":
        if target.ndim == 2:
            target = target[:, None, :]
        loss = cross_entropy_loss(output[:, -1, :], target[:, -1, :])
    else:
        raise ValueError(
            f"Unknown loss_aggregation: {loss_aggregation}, must be 'sum_of_softmax' or 'per_timestep_cross_entropy'"
        )
    return loss


def get_prediction(output, ignore_mask, prediction_mode):
    if prediction_mode == "sum_of_softmax":
        logits = masked_sum_of_softmaxes(output, ignore_mask)
    elif prediction_mode == "mean_over_sequence":
        logits = jnp.mean(output * ignore_mask[..., None], axis=1)
    elif prediction_mode == "per_timestep_prediction":
        logits = output
    elif prediction_mode == "last_timestep_prediction":
        logits = output[:, -1, :]
    else:
        raise ValueError(
            f"Unknown prediction_mode: {prediction_mode}, must be 'sum_of_softmax' or 'mean_over_sequence'"
        )
    return logits
