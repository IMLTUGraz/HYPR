from flax.nnx.nn.recurrent import RNNCellBase
import flax.nnx as nnx
import jax

from util.utils import get_initializer


class RNNBaseWithLinear(RNNCellBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_recurrent,
        kernel_initializer,
        rec_initializer,
        use_bias,
        rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.is_recurrent = is_recurrent
        fan_in = in_features + out_features if is_recurrent else in_features
        self.linear = nnx.Linear(
            in_features,
            out_features,
            rngs=rngs,
            kernel_init=get_initializer(kernel_initializer, fan_in, out_features),
            use_bias=use_bias,
        )
        # jax.debug.print("linear kernel shape: {}", self.linear.kernel.shape)
        if is_recurrent:
            # self.recurrent = nnx.Linear(out_features, out_features, rngs=rngs, use_bias=False, kernel_init=nnx.initializers.orthogonal())
            self.recurrent = nnx.Linear(
                out_features,
                out_features,
                rngs=rngs,
                use_bias=False,
                kernel_init=get_initializer(
                    rec_initializer, fan_in, out_features
                ),
            )
        else:
            self.recurrent = None

    @property
    def state_dim(self):
        raise NotImplementedError

    @property
    def get_params(self):
        return self.params

    def apply_parameter_constraints(self):
        pass

    def __call__(self, inputs, init_state):
        I = self.linear(inputs)
        last_state, output = self.multi_step_forward(self, I, init_state)
        return output, last_state

    @staticmethod
    @nnx.scan(in_axes=(None, 1, nnx.Carry), out_axes=(nnx.Carry, 1))
    def multi_step_forward(neuron, I, s):
        init_cell_state = s
        # new_state, (z, dz_ds) = state_transition_with_recurrence(I, s, self.params)
        if neuron.is_recurrent:
            # recurrent state
            # I_rec = memory_profiler_fn(neuron.recurrent(init_cell_state[..., -1]))
            I_rec = neuron.recurrent(init_cell_state[..., -1])
            I = I + I_rec
        new_state, z_with_aux = neuron.step_fn(
            I, init_cell_state, neuron.params, neuron.hyperparams
        )
        z = z_with_aux[0]
        # carry, output
        return new_state, z


# @nnx.scan(in_axes=(None, 1, nnx.Carry), out_axes=(nnx.Carry, 0))
# def multi_step_forward_no_rec(neuron, I, s):
#     new_state, (z, dz_ds) = adlif_hypr_state_transition(I, s, neuron.params)

#     # carry, output
#     return new_state, (new_state, z, dz_ds)


# @nnx.custom_vjp(nondiff_argnums=(nnx.DiffState(3, False),))
# def _generic_process_timeframe(neuron, x, s, elig_matrix):
#     I = neuron.linear(x)
#     last_state, (_, output, _) = multi_step_forward(neuron, I, s)
#     return (jnp.swapaxes(output, 0, 1), last_state, jnp.zeros_like(elig_matrix))
