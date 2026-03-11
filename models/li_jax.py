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

# class LIHypr(GenericHyprTrainableBaseRNNCell): 
#     def __init__(self, 
#                  in_features: int,
#                 out_features: int,
#                 is_recurrent,
#                 hyperparams,
#                 rngs,
#                 **kwargs):
#         super().__init__(in_features, out_features, is_recurrent, rngs, **kwargs)
#         self.hyperparams = hyperparams
#         self.params = nnx.Param({
#             "tau_u": sample_from_range([0.0, 1.0], (out_features,), rngs.params())
#         })

#     @property
#     def state_dim(self):
#         return 2

#     def initialize_carry(self, input_shape):
#         batch_size, feature_dim = input_shape
#         return jnp.zeros((batch_size, self.out_features, self.state_dim))
    
#     def apply_parameter_constraints(self):
#         self.params["tau_u"] = jnp.clip(self.params.value["tau_u"], 0.0, 1.0)
    
#     # must be static because of jax.jacfwd
#     @staticmethod
#     def step_fn(input_current, carry, params, hyperparams):        
#         dt = 1.0

#         tau_u_min, tau_u_max = hyperparams["tau_u_range"]

#         tau_u = tau_u_min + (tau_u_max - tau_u_min) * params["tau_u"]
#         alpha = jnp.exp(-dt / tau_u)
        
#         # Compute the new membrane potential
#         u_t = alpha * carry[..., 0] + (1.0 - alpha) * input_current   
        
#         # state, output
#         return jnp.stack([u_t, u_t], axis=-1), (u_t, jnp.ones_like(u_t[...,None]))
    

class LI(RNNBaseWithLinear): 
    def __init__(self, 
                 in_features: int,
                out_features: int,
                is_recurrent,
                kernel_initializer,
                rec_initializer,
                use_bias,
                hyperparams,
                rngs,
                **kwargs):
        super().__init__(in_features, out_features, is_recurrent, 
            kernel_initializer,
            rec_initializer, use_bias, rngs, **kwargs)
        self.hyperparams = hyperparams
        if hyperparams["interpolation_tau"]:
            self.params = nnx.Param({
                "tau_u": sample_from_range([0.0,1.0], (out_features,), rngs.params()),
            })
        else:
            self.params = nnx.Param({
                "tau_u": sample_from_range(hyperparams["tau_u_range"], (out_features,), rngs.params()),
            })

    @property
    def state_dim(self):
        return 2

    def initialize_carry(self, input_shape):
        batch_size, feature_dim = input_shape
        return jnp.zeros((batch_size, self.out_features, self.state_dim))
    
    def apply_parameter_constraints(self):
        if self.hyperparams["interpolation_tau"]:
            self.params["tau_u"] = jnp.clip(self.params.value["tau_u"], 0.0, 1.0)
        else:
            self.params["tau_u"] = jnp.clip(self.params.value["tau_u"], self.hyperparams["tau_u_range"][0], self.hyperparams["tau_u_range"][1])
    
    # must be static because of jax.jacfwd
    @staticmethod
    def step_fn(input_current, carry, params, hyperparams):        
        dt = 1.0
        if hyperparams["interpolation_tau"]:
            tau_u_min, tau_u_max = hyperparams["tau_u_range"]

            tau_u = tau_u_min + (tau_u_max - tau_u_min) * params["tau_u"]
        else:
            tau_u = params["tau_u"]
        # Compute decay factors (alpha and beta)
        alpha = jnp.exp(-dt / tau_u)
        
        # Compute the new membrane potential
        u_t = alpha * carry[..., 0] + (1.0 - alpha) * input_current   
        
        # state, (output, gradient of output wrt state)
        #TODO: remove dz_ds, use autodiff!
        return jnp.stack([u_t, u_t], axis=-1), (u_t, jnp.ones_like(u_t[...,None]))


