import jax
from flax import nnx
import jax.numpy as jnp
from jax import lax


def binary_assoc_operator(a, b):
    """binary associative operator for the scan (see Appendix G)

    Args:
        a: (num_neurons, state_dim, state_dim)    ds(t)/ds(t-1) matrix
        b: (num_neurons, state_dim, state_dim)    ds(t)/ds(t-1) matrix
    Returns:
        (num_neurons, state_dim, state_dim)    ds(t)/ds(t-1) matrix
        (num_neurons, state_dim)                dL/ds(t-1)
    """
    At, dLt_dst = b
    At1, dLt1_dst = a
    return At @ At1, jnp.matvec(At, dLt1_dst) + dLt_dst


# not used, but kept for understandability
def hypr_sequentially(
    elig_ff_t0,
    elig_rec_t0,
    elig_param_t0,
    elig_bias_t0,
    X,
    z,
    ds_dI,
    ds_dP,
    L,
    A_list: jnp.ndarray,
    recurrent,
):
    """Sequential forward eligibility propagation for a single time step.

    Args:
        elig_t0: (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (from previous timeframe)
        X:      (seq_len, n_input_neurons)                    Input matrix, inputs per time step
        ds_dI:  (seq_len, num_neurons, state_dim)             Derivative of the state w.r.t. the input
        L:      (seq_len, num_neurons, state_dim)             (Partial, not total) derivative of the loss at each time step w.r.t. the state
        A_list: (num_neurons, seq_len, state_dim, state_dim)  List of ds(t)/ds(t-1) matrices for all time steps t
    Returns:
        e_tau:  (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (for the next timeframe)
    """

    A_list = jnp.swapaxes(A_list, 1,2)

    @nnx.scan(in_axes=(1, 1, 1, 1, 1, 1, nnx.Carry), out_axes=(1, nnx.Carry))
    def scan_fn(A, ds_dI, ds_dp, z, x, l, carry):
        elig_ff, elig_rec, elig_param, elig_bias = carry
        # elig_ff = A @ elig_ff + ds_dI @ X

        elig_ff = jnp.einsum("bnji,bnmi->bnmj", A, elig_ff) + jnp.einsum(
            "bni,bm->bnmi", ds_dI, x
        )
        
        delta_w = jnp.einsum("bnmi,bni->bmn", elig_ff, l)

        # if recurrent:
        #     elig_rec = jnp.einsum("nji,nmi->nmj", A, elig_rec) + jnp.einsum(
        #         "ni,m->nmi", ds_dI, z
        #     )
        # else:
        #     elig_rec = None

        # elig_param_new = jax.tree_util.tree_map(
        #     lambda x, y: jnp.einsum("nji,ni->nj", A, x) + y,
        #     elig_param,
        #     ds_dp,
        # )

        carry_new = (elig_ff, elig_rec, elig_param, elig_bias)
        return (carry_new, delta_w), carry_new

    carry_init = (elig_ff_t0, elig_rec_t0, elig_param_t0, elig_bias_t0)
    _, ((e_ff_tau, e_rec_tau, e_param_tau, e_bias_tau), delta_w) = scan_fn(
        A_list[..., :-1, :, :, :], ds_dI, ds_dP, z, X, L, carry_init
    )

    return e_ff_tau, e_rec_tau, e_param_tau, e_bias_tau, delta_w

def hypr_backprop_assoc(
    elig_ff_t0,
    elig_rec_t0,
    elig_p_t0,
    elig_bias_t0,
    z_list,
    X,
    ds_dI,
    ds_dP,
    L,
    A_list,
    recurrent,
):
    """Core associative hypr

    Args:
        elig_t0: (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (from previous timeframe)
        X:      (seq_len, n_input_neurons)                    Input matrix, inputs per time step
        ds_dI:  (seq_len, num_neurons, state_dim)             Derivative of the state w.r.t. the input
        L:      (seq_len, num_neurons, state_dim)             (Partial, not total) derivative of the loss at each time step w.r.t. the state
        A_list: (num_neurons, seq_len, state_dim, state_dim)  List of ds(t)/ds(t-1) matrices for all time steps t
    Returns:
        dL_dW: (num_neurons, n_input_neurons)                dL/dW
    """

    A_list = jnp.swapaxes(A_list, 1, 2)  # seq first


    # dL_ds is called q in the paper. It is not the "true" dL_ds (see Appendix F)
    _, dL_ds = jax.lax.associative_scan(
        binary_assoc_operator, (A_list.mT, L), axis=1, reverse=True
    )

    dL_dI = jnp.einsum("bsni,bsni->bsn", dL_ds[..., 1:, :, :], ds_dI)
    dL_dW_ff = jnp.einsum("bsn,bsm->nm", dL_dI, X) + jnp.einsum(
        "bni,bnmi->nm", dL_ds[..., 0, :, :], elig_ff_t0
    )
    if recurrent:
        dL_dW_rec = jnp.einsum("bsn,bsm->nm", dL_dI, z_list) + jnp.einsum(
            "bni,bnmi->nm", dL_ds[..., 0, :, :], elig_rec_t0
        )
    else:
        dL_dW_rec = None

    dL_dP = jax.tree_util.tree_map(
        lambda x, y: jnp.einsum("bsni,bsni->n", dL_ds[..., 1:, :, :], x)
        + jnp.einsum("bni,bni->n", dL_ds[..., 0, :, :], y),
        ds_dP,
        elig_p_t0,
    )
    dL_dbias = jnp.einsum("bsn->n", dL_dI) + jnp.einsum(
        "bni,bni->n", dL_ds[..., 0, :, :], elig_bias_t0
    )
    return dL_dW_ff, dL_dW_rec, dL_dP, dL_dbias


def fast_forward_elig(
    elig_ff_t0,
    elig_rec_t0,
    elig_param_t0,
    elig_bias_t0,
    X,
    z,
    ds_dI,
    ds_dP,
    A_list: jnp.ndarray,
    recurrent,
):
    """Compute the eligibility matrix at the last time step of the subsequence to pass on to the next subsequence.

    Args:
        elig_t0: (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (from previous timeframe)
        X:      (seq_len, n_input_neurons)                    Input matrix, inputs per time step
        ds_dI:  (seq_len, num_neurons, state_dim)             Derivative of the state w.r.t. the input
        L:      (seq_len, num_neurons, state_dim)             (Partial, not total) derivative of the loss at each time step w.r.t. the state
        A_list: (num_neurons, seq_len, state_dim, state_dim)  List of ds(t)/ds(t-1) matrices for all time steps t
    Returns:
        e_tau:  (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (for the next timeframe)
    """

    P = lax.associative_scan(
        jnp.matmul, A_list, axis=1, reverse=True
    ).mT  # P: [At * ... * A1,...,At]

    # # Compute the eligibility matrix at the last time step to pass on to the next timeframe

    e_ff_tau = elig_ff_t0 @ P[..., 0, :, :] + jnp.einsum(
        "nsij,sni,sm->nmj", P[..., 1:, :, :], ds_dI, X
    )
    e_rec_tau = (
        elig_rec_t0 @ P[..., 0, :, :]
        + jnp.einsum("nsij,sni,sm->nmj", P[..., 1:, :, :], ds_dI, z)
        if recurrent
        else None
    )
    e_bias_tau = jnp.vecmat(elig_bias_t0, P[..., 0, :, :]) + jnp.einsum(
        "nsij,sni->nj", P[..., 1:, :, :], ds_dI
    )
    e_param_tau = jax.tree_util.tree_map(
        lambda e_p, ds_dp: jnp.vecmat(e_p, P[..., 0, :, :])
        + jnp.einsum("nsij,sni->nj", P[..., 1:, :, :], ds_dp),
        elig_param_t0,
        ds_dP,
    )

    return e_ff_tau, e_rec_tau, e_param_tau, e_bias_tau

# never used but kept for understandability.
# This is the sequential (slow) version of calculating the elig. matrix.
def fast_forward_elig_sequentially(
    elig_ff_t0,
    elig_rec_t0,
    elig_param_t0,
    elig_bias_t0,
    X,
    z,
    ds_dI,
    ds_dP,
    A_list: jnp.ndarray,
    recurrent,
):
    """Fast forward eligibility propagation for a single time step.

    Args:
        elig_t0: (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (from previous timeframe)
        X:      (seq_len, n_input_neurons)                    Input matrix, inputs per time step
        ds_dI:  (seq_len, num_neurons, state_dim)             Derivative of the state w.r.t. the input
        L:      (seq_len, num_neurons, state_dim)             (Partial, not total) derivative of the loss at each time step w.r.t. the state
        A_list: (num_neurons, seq_len, state_dim, state_dim)  List of ds(t)/ds(t-1) matrices for all time steps t
    Returns:
        e_tau:  (num_neurons, n_input_neurons, state_dim)    Eligibility matrix at time t=-1 (for the next timeframe)
    """

    A_list = jnp.swapaxes(A_list, 0, 1)

    @nnx.scan(in_axes=(0, 0, 0, 0, 0, nnx.Carry), out_axes=(0, nnx.Carry))
    def scan_fn(A, ds_dI, ds_dp, z, x, carry):
        elig_ff, elig_rec, elig_param, elig_bias = carry
        # elig_ff = A @ elig_ff + ds_dI @ X

        elig_ff = jnp.einsum("nji,nmi->nmj", A, elig_ff) + jnp.einsum(
            "ni,m->nmi", ds_dI, x
        )

        if recurrent:
            elig_rec = jnp.einsum("nji,nmi->nmj", A, elig_rec) + jnp.einsum(
                "ni,m->nmi", ds_dI, z
            )
        else:
            elig_rec = None

        elig_param_new = jax.tree_util.tree_map(
            lambda x, y: jnp.einsum("nji,ni->nj", A, x) + y,
            elig_param,
            ds_dp,
        )

        carry_new = (elig_ff, elig_rec, elig_param_new, elig_bias)
        return carry_new, carry_new

    carry_init = (elig_ff_t0, elig_rec_t0, elig_param_t0, elig_bias_t0)
    _, (e_ff_tau, e_rec_tau, e_param_tau, e_bias_tau) = scan_fn(
        A_list[..., :-1, :, :, :], ds_dI, ds_dP, z, X, carry_init
    )

    return e_ff_tau, e_rec_tau, e_param_tau, e_bias_tau


def batched_fast_forward_elig(
    elig_ff_t0,
    elig_rec_t0,
    elig_param_t0,
    elig_bias_t0,
    X,
    z,
    ds_dI,
    ds_dP,
    A_list,
    recurrent,
):
    return jax.vmap(fast_forward_elig, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))(
        elig_ff_t0,
        elig_rec_t0,
        elig_param_t0,
        elig_bias_t0,
        X,
        z,
        ds_dI,
        ds_dP,
        A_list,
        recurrent,
    )
