import flax.nnx as nnx
from models.rnn_base import RNNBaseWithLinear
import jax
import jax.numpy as jnp

class MultiLayerRNN(nnx.Module):
    def __init__(
        self,
        hidden_layer_class : RNNBaseWithLinear,
        hidden_layer_params,
        hidden_layer_dim,
        hidden_layer_recurrent,
        hidden_layer_kernel_initializer,
        hidden_layer_rec_initializer,
        hidden_layer_bias,
        num_hidden_layers,
        output_layer_class : RNNBaseWithLinear | nnx.Module,
        output_layer_params,
        output_layer_initializer,
        input_dim,
        output_dim,
        rngs
    ):
        super(MultiLayerRNN, self).__init__()
        layers = []
        input_shapes = []
        self.n_layers = num_hidden_layers + 1

        self.hidden_layer_size= hidden_layer_dim

        for n in range(num_hidden_layers):
            in_dim = input_dim if n == 0 else hidden_layer_dim
            layer = hidden_layer_class(
                in_features = in_dim,
                out_features = hidden_layer_dim,
                is_recurrent = hidden_layer_recurrent,
                hyperparams = hidden_layer_params,
                kernel_initializer = hidden_layer_kernel_initializer,
                rec_initializer = hidden_layer_rec_initializer,       
                use_bias = hidden_layer_bias,        
                rngs = rngs
            )
            layers.append(layer)
            input_shapes.append(in_dim)

        self.output_layer = output_layer_class(
            in_features = hidden_layer_dim,
            out_features = output_dim,
            is_recurrent = False,
            hyperparams = output_layer_params,
            kernel_initializer = output_layer_initializer,
            rec_initializer = None,
            use_bias = True,
            rngs = rngs
        )
        input_shapes.append(hidden_layer_dim)
        self.layers = tuple(layers)
        self.input_shapes = tuple(input_shapes)

    def initialize_carry(self, input_shape):
        batch_size, feature_dim = input_shape
        # initial states
        init_states = [
            l.initialize_carry((batch_size, in_shape)) for l,in_shape in zip(self.layers, self.input_shapes) 
            ]
        init_states.append(self.output_layer.initialize_carry((batch_size, self.input_shapes[-1])))
        return init_states
    
    def apply_parameter_constraints(self):
        for layer in self.layers:
            layer.apply_parameter_constraints()

    
    def __call__(self, X, state, return_spike_rates=False):
        return self.scan_layers(X, state, return_spike_rates)


    def scan_layers(self, x, state, return_spike_rates):
        if return_spike_rates:
            spike_rates = []
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, state[i])
            state[i] = new_state
            if return_spike_rates:
                spike_rates.append(jnp.mean(x))
        out, new_out_state = self.output_layer(x, state[-1])
        state[-1] = new_out_state
        if return_spike_rates:
            return (out, spike_rates), state
        return out, state
