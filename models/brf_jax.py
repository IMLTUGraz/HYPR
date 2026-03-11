import jax
import jax.numpy as jnp
from functools import partial
from models.rnn_base import RNNBaseWithLinear
from util.surrogate_derivative import spike_double_gaussian_FGI, spike_slayer_FGI, surrogate_gradient_FGI
import jax.nn.initializers as init
from jax import random
import flax.nnx as nnx
from util.utils import sample_from_range


class BRF(RNNBaseWithLinear):
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
        super().__init__(in_features, out_features, is_recurrent,kernel_initializer,
            rec_initializer, use_bias, rngs, **kwargs)

        self.hyperparams = hyperparams

        b_range = hyperparams["b_range"] if not hyperparams["exponential_b"] else jnp.log(jnp.array(hyperparams["b_range"]) + 1e-10)
        omega_range = hyperparams["omega_range"] if not hyperparams["exponential_omega"] else jnp.log(jnp.array(hyperparams["omega_range"]) + 1e-10)

        self.params = nnx.Param(
            {
            "omega": sample_from_range(omega_range, (out_features,), rngs.params()),
            "b_offset": sample_from_range(b_range, (out_features,), rngs.params()),
           }
        )

    @property
    def state_dim(self):
        return 4

    def initialize_carry(self, input_shape, rngs=None):
        batch_size, feature_dim = input_shape
        return jnp.zeros((batch_size, self.out_features, self.state_dim))


    def apply_parameter_constraints(self):
        if self.hyperparams["exponential_omega"]:
            self.params["omega"] = jnp.clip(self.params.value["omega"], -10.0, jnp.log(jnp.pi/self.hyperparams["dt"]))





    @staticmethod
    def step_fn(input_current, carry, params, hyperparams):
        u_tm1, v_tm1, q_tm1 = carry[..., 0], carry[..., 1], carry[..., 2]

        use_exponential = hyperparams["use_exponential"]
        dt = hyperparams["dt"]
        if hyperparams["exponential_b"]:
            b_offset = jnp.exp(params["b_offset"])
        else:
            b_offset = jnp.abs(params["b_offset"])

        if hyperparams["exponential_omega"]:
            omega = jnp.exp(params["omega"])
        else:
            omega = jnp.abs(params["omega"])
        theta = hyperparams["thr"]
        p_omega = (-1 + jnp.sqrt(1 - jnp.square(dt * omega))) / dt

        if use_exponential:
            b = - b_offset - q_tm1
            s_t = jnp.exp(dt * (b + omega * 1j)) * (u_tm1 + 1j * v_tm1) 
            u_t = jnp.real(s_t) + dt * input_current
            v_t = jnp.imag(s_t)
        else:
            b = p_omega - b_offset - q_tm1
            u_t = u_tm1 + b * u_tm1 * dt - omega * v_tm1 * dt + input_current * dt
            v_t = v_tm1 + omega * u_tm1 * dt + b * v_tm1 * dt

        z_raw = u_t - theta - q_tm1

        z_t, d_z_d_u = surrogate_gradient_FGI(z_raw, hyperparams["surrogate_gradient"])


        q_t = 0.9 * q_tm1 + z_t

        d_z_d_v = jax.lax.stop_gradient(d_z_d_u) * 0.0
        d_z_d_ref = d_z_d_u * -1

        new_state = jnp.stack([u_t, v_t, q_t, z_t], axis=-1)
        grads = jnp.stack([d_z_d_u, d_z_d_v, d_z_d_ref], axis=-1)

        return new_state, (z_t, grads)
