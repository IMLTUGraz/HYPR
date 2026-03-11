import jax
import jax.numpy as jnp
from functools import partial
from models.rnn_base import RNNBaseWithLinear

# from hypr_trainable_base import GenericHyprTrainableBaseRNNCell
from util.surrogate_derivative import spike_slayer_FGI

import jax.nn.initializers as init
from jax import random

import flax.nnx as nnx
from util.utils import sample_from_range

class AdLIF(RNNBaseWithLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_recurrent,
        kernel_initializer,
        rec_initializer,
        use_bias,
        hyperparams,
        rngs,
        **kwargs,
    ):
        super().__init__(
            in_features,
            out_features,
            is_recurrent,
            kernel_initializer,
            rec_initializer,
            use_bias,
            rngs,
            **kwargs,
        )
        self.hyperparams = hyperparams

        self.params = nnx.Param(
            {
                "tau_u": sample_from_range([0.0, 1.0], (out_features,), rngs.params()),
                "tau_w": sample_from_range([0.0, 1.0], (out_features,), rngs.params()),
                "a": sample_from_range(
                    hyperparams["a_range"], (out_features,), rngs.params()
                ),
                "b": sample_from_range(
                    hyperparams["b_range"], (out_features,), rngs.params()
                ),
                "thr": jnp.full((out_features,), hyperparams["thr"]),
            }
        )
        # super().__init__(in_features, out_features, is_recurrent, rngs, **kwargs)

    @property
    def state_dim(self):
        return 3

    def initialize_carry(self, input_shape):
        batch_size, feature_dim = input_shape
        return jnp.zeros((batch_size, self.out_features, self.state_dim))

    def apply_parameter_constraints(self):
        self.params["a"] = jnp.clip(
            self.params.value["a"],
            self.hyperparams["a_range"][0],
            self.hyperparams["a_range"][1],
        )
        self.params["b"] = jnp.clip(
            self.params.value["b"],
            self.hyperparams["b_range"][0],
            self.hyperparams["b_range"][1],
        )
        # self.params["tau_u"] = jnp.clip(self.params.value["tau_u"], self.hyperparams["tau_u_range"][0], self.hyperparams["tau_u_range"][1])
        # self.params["tau_w"] = jnp.clip(self.params.value["tau_w"], self.hyperparams["tau_w_range"][0], self.hyperparams["tau_w_range"][1])
        self.params["tau_u"] = jnp.clip(self.params.value["tau_u"], 0.0, 1.0)
        self.params["tau_w"] = jnp.clip(self.params.value["tau_w"], 0.0, 1.0)

    # must be static because of jax.jacfwd
    @staticmethod
    def step_fn(input_current, carry, params, hyperparams):
        # params = jax.lax.stop_gradient(params)
        # Compute last time-step spike; use stop_gradient to avoid backpropagating through it.

        # my_z_tm1 = jax.lax.stop_gradient(
        #     jnp.heaviside(carry[..., 0] - params["thr"], 0.0)
        # )
        
        # previous spike output
        my_z_tm1 = carry[..., 2]

        dt = 1.0

        tau_u_min, tau_u_max = hyperparams["tau_u_range"]
        tau_w_min, tau_w_max = hyperparams["tau_w_range"]

        tau_u = tau_u_min + (tau_u_max - tau_u_min) * params["tau_u"]
        tau_w = tau_w_min + (tau_w_max - tau_w_min) * params["tau_w"]

        # Compute decay factors (alpha and beta)
        alpha = jnp.exp(-dt / tau_u)
        beta = jnp.exp(-dt / tau_w)

        # Compute the new membrane potential
        u_t = alpha * carry[..., 0] * (1.0 - my_z_tm1) + (1.0 - alpha) * (
            input_current - carry[..., 1]
        )

        pre_spike = u_t - jax.lax.stop_gradient(params["thr"])
        # Get spike output and surrogate gradient for u
        z_t, d_z_d_pre_spike = spike_slayer_FGI(pre_spike, 5, 0.2)

        d_z_d_u = d_z_d_pre_spike

        # Update adaptation carry
        w_t = (
            beta * carry[..., 1]
            + (1.0 - beta)
            * (params["a"] * u_t * (1.0 - z_t) + params["b"] * my_z_tm1)
            * hyperparams["adapt_coeff"]
        )
        # Dummy gradient for adaptation carry w (set to zero)
        d_z_d_w = jax.lax.stop_gradient(d_z_d_u) * 0.0

        # state, output
        return jnp.stack([u_t, w_t, z_t], axis=-1), (
            z_t,
            jnp.stack([d_z_d_u, d_z_d_w], axis=-1),
        )
