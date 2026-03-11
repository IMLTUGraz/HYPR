import jax
import jax.numpy as jnp
from functools import partial
from models.rnn_base import RNNBaseWithLinear
from util.surrogate_derivative import spike_slayer_FGI
import jax.nn.initializers as init
from jax import random
import flax.nnx as nnx
from util.utils import sample_from_range, sample_from_gaussian

class ALIF(RNNBaseWithLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_recurrent: bool,
        kernel_initializer,
        rec_initializer,
        use_bias: bool,
        hyperparams: dict,
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
                "tau_u": sample_from_gaussian(hyperparams["tau_u_mean"],hyperparams["tau_u_std"], (out_features,), rngs.params()),
                "tau_a": sample_from_gaussian(hyperparams["tau_a_mean"],hyperparams["tau_a_std"], (out_features,), rngs.params())
            }
        )

    @property
    def state_dim(self) -> int:
        return 3  # [u, a, z]

    def initialize_carry(self, input_shape):
        batch_size, _ = input_shape
        return jnp.zeros((batch_size, self.out_features, self.state_dim))

    def apply_parameter_constraints(self):
        self.params["tau_u"] = jnp.clip(self.params.value["tau_u"], min=10.0)
        self.params["tau_a"] = jnp.clip(self.params.value["tau_a"], min=100.0)
        # pass

    @staticmethod
    def step_fn(input_current, carry, params, hyperparams):
        u_tm1 = carry[..., 0]  # membrane potential
        a_tm1 = carry[..., 1]  # adaptation variable
        z_tm1 = carry[..., 2]  # spike variable

        # A_tm1 = hyperparams["b_j0"] + hyperparams["beta"] * a_tm1
        
        dt = 1.0

        # tau_u_min, tau_u_max = hyperparams["tau_u_range"]
        # tau_u = tau_u_min + (tau_u_max - tau_u_min) * params["tau_u"]

        tau_u = params["tau_u"]
        alpha = jnp.exp(-dt / tau_u)

        # tau_a_min, tau_a_max = hyperparams["tau_a_range"]
        # tau_a = tau_a_min + (tau_a_max - tau_a_min) * params["tau_a"]
        tau_a = params["tau_a"]
        rho = jnp.exp(-dt / tau_a)

        # Update membrane potential
        #u_t = alpha * u_tm1 * (1.0 - z_tm1) + (1.0 - alpha) * input_current
        a_t = rho * a_tm1 + (1 - rho) * z_tm1
        A = hyperparams["b_j0"] + hyperparams["beta"] * a_t

        u_t = alpha * u_tm1 + (1 - alpha)  * input_current - A * z_tm1 * dt

        # # Update adaptation variable
        # a_t = rho * a_tm1 + (1 - rho) * z_tm1

        # A_t = hyperparams["b_j0"] + hyperparams["beta"] * a_t

        # z_raw = u_t - A_t
        z_raw = u_t - A

        z_t, d_z_d_zraw = spike_slayer_FGI(z_raw, 5, 0.2)
        # If you want to switch to Gaussian surrogate:
        # z_t, d_z_d_zraw = spike_double_gaussian_FGI_ALIF_ORIG(z_raw)

        d_z_d_u = d_z_d_zraw
        d_z_d_a = d_z_d_zraw * hyperparams["beta"] * -1 # because d_zraw_d_a = -beta

        return jnp.stack([u_t, a_t, z_t], axis=-1), (
            z_t,
            jnp.stack([d_z_d_u, d_z_d_a], axis=-1),
        )
