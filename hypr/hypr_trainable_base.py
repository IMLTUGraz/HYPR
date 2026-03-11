import time
import jax
import jax.numpy as jnp
from flax import nnx
from collections import namedtuple
from hypr.hypr_helpers import batched_fast_forward_elig, hypr_backprop_assoc
import sys
from models.rnn_base import RNNBaseWithLinear

FF_KERNEL_KEY = "linear.kernel"
FF_BIAS_KEY = "linear.bias"
REC_WEIGHTS_KEY = "recurrent.kernel"
PARAM_KEY = "params"


def maybe_hypr_class_factory(cell_class, use_hypr, single_step=False):
    """
    Factory function to create a class that wraps a given cell class with eligibility propagation functionality.
    """

    class HyprWrapper(cell_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batched_jacobian = ew_get_jacobian(self)

        def __call__(self, x, s):
            init_state, elig_matrix = s
            output, last_state, elig_matrix = ew_forward(
                self, x, init_state, elig_matrix
            )
            return output, (last_state, elig_matrix)

        def initialize_carry(self, input_shape):
            batch_size, feature_dim = input_shape
            init_state = super().initialize_carry((batch_size, feature_dim))
            elig_matrix = ew_initialize_elig_matrix(self, (batch_size, feature_dim))
            return (init_state, elig_matrix)

    return HyprWrapper if use_hypr or single_step else cell_class


def ew_get_jacobian(cell):
    full_jacobian = jax.jacfwd(cell.step_fn, argnums=(0, 1, 2), has_aux=True)
    vmap_cells = nnx.vmap(full_jacobian, in_axes=(0, 0, 0, None))
    # Next, create a vmap over the subsequence dimension:
    vmap_subsequence = nnx.vmap(vmap_cells, in_axes=(0, 0, None, None))
    # Finally, create a vmap over the batch dimension:
    vmap_batch = nnx.vmap(vmap_subsequence, in_axes=(0, 0, None, None))
    return vmap_batch


def ew_initialize_elig_matrix(cell, input_shape):
    batch_size, feature_dim = input_shape
    elig_per_param = jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (batch_size,) + x.shape + (cell.state_dim - 1,), dtype=x.dtype
        ),
        cell.params.value,
    )
    return {
        FF_KERNEL_KEY: jnp.zeros(
            (batch_size, cell.out_features, feature_dim, cell.state_dim - 1)
        ),  # because last state is output
        FF_BIAS_KEY: jnp.zeros((batch_size, cell.out_features, cell.state_dim - 1)),
        REC_WEIGHTS_KEY: jnp.zeros(
            (batch_size, cell.out_features, cell.out_features, cell.state_dim - 1)
        )
        if cell.is_recurrent
        else None,
        PARAM_KEY: elig_per_param,
    }


@nnx.scan(in_axes=(None, 1, nnx.Carry), out_axes=(nnx.Carry, 1))
def multi_step_forward_with_states(cell, I, s):
    if cell.is_recurrent:
        # recurrent state
        I_rec = cell.recurrent(s[..., -1])
        I = I + I_rec
    new_state, (z, dz_ds) = cell.step_fn(I, s, cell.params, cell.hyperparams)
    # carry, output
    return new_state, (new_state, z, dz_ds)


def ew_forward(cell, x, s, elig_matrix):
    return _generic_process_timeframe(cell, x, s, elig_matrix)


@nnx.custom_vjp(nondiff_argnums=(nnx.DiffState(3, False),))
def _generic_process_timeframe(cell, x, state, elig_matrix):
    I = cell.linear(x)
    last_state, (_, output, _) = multi_step_forward_with_states(cell, I, state)
    return (output, last_state, elig_matrix)


def _generic_hypr_forward(cell, x, s0, elig_matrix):
    ff_elig, rec_elig, param_elig, bias_elig = (
        elig_matrix[FF_KERNEL_KEY],
        elig_matrix[REC_WEIGHTS_KEY],
        elig_matrix[PARAM_KEY],
        elig_matrix[FF_BIAS_KEY],
    )

    # parallel neuron input
    I = cell.linear(x)

    ##### S-stage ####
    last_state, (state_list, output_list, dz_ds_list) = multi_step_forward_with_states(
        cell, I, s0
    )
    state_dim = state_list.shape[-1]

    if cell.is_recurrent:
        I_rec = cell.recurrent(state_list[..., -1])
    else:
        I_rec = 0.0

    ##### P-stage ####

    # add last state of previous time frame to state list
    state_list = jnp.concatenate([s0[..., None, :, :], state_list], 1)

    # compute gradients
    (ds_dI, ds_ds, ds_dP), aux = cell.batched_jacobian(
        I + I_rec, state_list[..., :-1, :, :], cell.params.value, cell.hyperparams
    )


    # append a dummy ds_ds for the last state
    dummy_ds_ds = jnp.broadcast_to(
        jnp.eye(state_dim)[None, None, None, :, :], ds_ds[..., -1:, :, :, :].shape
    )
    ds_ds = jnp.concatenate([ds_ds, dummy_ds_ds], -4)

    # remove z from the gradients, since z is only needed for the recurrence
    ds_ds = ds_ds[..., :-1, :-1]
    ds_dI = ds_dI[..., :-1]
    ds_dP = jax.tree_util.tree_map(lambda x: x[..., :-1], ds_dP)

    # compute eligibility matrix at t=lambda
    ff_elig_new, rec_elig_new, param_elig_new, bias_elig_new = (
        batched_fast_forward_elig(
            ff_elig,
            rec_elig,
            param_elig,
            bias_elig,
            x,
            output_list,
            ds_dI,
            ds_dP,
            jnp.swapaxes(ds_ds, 1, 2),
            recurrent=cell.is_recurrent,
        )
    )

    # residual is passed to the backward function
    residual = (
        ds_dI,
        ds_ds,
        ds_dP,
        x,
        output_list,
        dz_ds_list,
        elig_matrix,
        cell,
    )

    # rebuild new eligibility matrix e_lambda
    elig_new = {
        FF_KERNEL_KEY: ff_elig_new,
        # FF_KERNEL_KEY: ff_elig,
        REC_WEIGHTS_KEY: rec_elig_new,
        # REC_WEIGHTS_KEY: rec_elig,
        PARAM_KEY: param_elig_new,
        FF_BIAS_KEY: bias_elig_new,
    }
    return ((output_list, last_state, elig_new), residual)


# @nnx.jit
def _generic_hypr_backward(residual, g):
    
    #### still P-stage ####
    (
        ds_dI,
        ds_ds,
        ds_dP,
        X,
        output_list,
        dz_ds_list,
        elig_matrix,
        cell,
    ) = residual

    # decompose eligibility matrix
    ff_elig, rec_elig, param_elig, bias_elig = (
        elig_matrix[FF_KERNEL_KEY],
        elig_matrix[REC_WEIGHTS_KEY],
        elig_matrix[PARAM_KEY],
        elig_matrix[FF_BIAS_KEY],
    )

    input_updates_g, out_g = g
    dLdz, _, _ = out_g
    (g_cell, _, _, g_elig_matrix) = input_updates_g
    m_g = jax.tree.map(lambda x: x, g_cell)  # create copy

    # only partial per-timestep derivatives, no backpropagation through time happening here
    dL_ds = dLdz[..., None] * dz_ds_list

    # set dummy dL0_ds0 = 0 (see Appendix G)
    dL_ds = jnp.concatenate([jnp.zeros_like(dL_ds[..., 0:1, :, :]), dL_ds], -3)

    # This is the backward SSM from Eq. (15) and Appendix G
    delta_w_ff, delta_w_rec, delta_p, delta_bias = hypr_backprop_assoc(
        ff_elig,
        rec_elig,
        param_elig,
        bias_elig,
        output_list,
        X,
        ds_dI,
        ds_dP,
        dL_ds,
        jnp.swapaxes(ds_ds, 1, 2),
        recurrent=cell.is_recurrent,
    )

    # Assign gradients for the optimizer
    m_g["linear"]["kernel"].value = delta_w_ff.mT
    if cell.is_recurrent:
        m_g["recurrent"]["kernel"].value = delta_w_rec.mT
    m_g["linear"]["bias"].value = delta_bias
    m_g["params"].value = delta_p

    # We still use SPATIAL backpropagation! Hence return proper spatial gradients (see Appendix I)
    dL_dX = jnp.einsum("bsni,bsni->bsn", dL_ds[:, 1:], ds_dI) @ cell.linear.kernel.T

    # g_elig_matrix contains None everywhere, hence no further differentiation
    return m_g, dL_dX, None, g_elig_matrix


_generic_process_timeframe.defvjp(_generic_hypr_forward, _generic_hypr_backward)
