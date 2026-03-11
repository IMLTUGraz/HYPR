import sys
import time
import pickle  # save best model
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
import tqdm
import traceback
from models.multi_layer_rnn import MultiLayerRNN
from flax import nnx
from train import eval_model, eval_model_online, train_batch_hypr, train_batch_bptt
from hypr.hypr_trainable_base import maybe_hypr_class_factory


from data.dataloader import load_dataset
from util.factories import get_model_class, optimizer_factory
from util.factories import lr_scheduler_factory
import random

jax.config.update('jax_compiler_enable_remat_pass', False)


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    # This main is used to circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664
    # It will print the proper error message to the submitit log file
    try:
        actual_main(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


def actual_main(cfg: DictConfig):
    # sleep for random time to avoid all jobs starting exactly at the same time
    time.sleep(random.uniform(0.0, 5.0))
    # Generate dataset
    key = jax.random.PRNGKey(cfg.seed)
    print(f"Using hypr: {cfg.training.hypr}")

    if cfg.get("debug", False):
        print("Debug mode enabled!! If you don't want this, set debug=False in main.yaml")
        jax.config.update("jax_disable_jit", True)
    
    if cfg.search_for_nan:
        print("Searching for NaN values in the model")
        jax.config.update("jax_debug_nans", True)


    test_dl, val_dl, train_dl, input_dim, total_seq_len, batch_size, num_classes = load_dataset(
        cfg.data_dir, cfg.dataset, key
    )
    print('dataset loaded:', cfg.dataset['dataset_name'])
    print('batch size:', batch_size)    
    # plot single sample

    #inputs_shape: (batch_size, seq_len, input_dim)
    if cfg.plot_a_datasample:
        import plotly.graph_objects as go
        sample = next(iter(train_dl))
        inputs, targets = sample
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=inputs[0, :, :].T,
            colorscale="Viridis",
            colorbar=dict(title="Spike Rate"),
        ))
        fig.show()
    
    rngs = nnx.Rngs(cfg.seed)
    
    model = MultiLayerRNN(
        maybe_hypr_class_factory(get_model_class(cfg.model.hidden_layer_class), cfg.training.hypr, single_step=False),
        cfg.model.hidden_layer_hyperparams,
        cfg.model.hidden_size,
        cfg.model.hidden_layer_recurrent,
        cfg.model.hidden_layer_kernel_initializer,
        cfg.model.hidden_layer_rec_initializer,
        cfg.model.hidden_layer_bias,
        cfg.model.num_layers,
        maybe_hypr_class_factory(get_model_class(cfg.model.output_layer_class), cfg.training.hypr, single_step=False),
        cfg.model.output_layer_params,
        cfg.model.output_layer_initializer,
        input_dim,
        num_classes,
        rngs,
    )

    num_params = sum([x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))])
    print(f"Number of Model Parameters: {num_params}")
    print('hidden size:', cfg.model.hidden_size)
    
    steps_per_epoch = len(train_dl)
    grads_sum_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), nnx.state(model, nnx.Param)
    )
    
    lr_scheduler = lr_scheduler_factory(
        cfg.training.learning_rate,
        steps_per_epoch,
        cfg.lr_scheduler)
    opt = optimizer_factory(
        cfg.optimizer,
        lr_scheduler)
    optimizer = nnx.Optimizer(
        model, opt
    )

    best_accuracy = 0.0
    patience = (
        cfg.training.patience
    )  # define in config or set manually for early stopping
    patience_counter = 0

    train_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average()
    )
    val_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average()
    )


    for epoch in range(cfg.training.epochs):

        epoch_time = time.time()
        # print(len(train_dl))

        model.train()
        pbar = tqdm.tqdm(train_dl, leave=False) # progress bar
        if epoch > 0:
            pbar.set_description(
                f"Epoch {epoch + 1}/{cfg.training.epochs} - Prev Loss: {val_metrics.compute()['loss']:.4f} - Prev Val Acc: {val_metrics.compute()['accuracy']:.4f} - Train Acc: {train_metrics.compute()['accuracy']:.4f} - Train Loss: {train_metrics.compute()['loss']:.4f}"
            )
        else:
            pbar.set_description(
                f"Epoch {epoch + 1}/{cfg.training.epochs}"
            )
        train_metrics.reset()
        for batch in pbar:
            # Unpack the batch
            inputs, targets = batch
            init_state = model.initialize_carry((inputs.shape[0], input_dim))

            ignore_mask = jnp.concatenate(
                [
                    jnp.zeros((inputs.shape[0], cfg.dataset.ignore_first_n_timesteps)),
                    jnp.ones_like(inputs)[:, cfg.dataset.ignore_first_n_timesteps :, 0],
                ],
                axis=1,
            )
            # Reshape the inputs to match the expected shape
            if cfg.training.hypr:
                inputs = inputs.reshape(
                    batch_size,
                    cfg.hypr_args.num_chunks,
                    jnp.maximum(total_seq_len // cfg.hypr_args.num_chunks, 1),
                    input_dim,
                )
                ignore_mask = ignore_mask.reshape(
                    batch_size,
                    cfg.hypr_args.num_chunks,
                    jnp.maximum(total_seq_len // cfg.hypr_args.num_chunks, 1),
                )


            if cfg.training.hypr:                
                loss, grads = train_batch_hypr(
                    model,
                    inputs,  # Input chunked batch
                    # labels[i],
                    targets,
                    init_state,
                    grads_sum_init,
                    optimizer,
                    train_metrics,
                    cfg.training.grad_clip_val,
                    ignore_mask,
                    cfg.training.prediction_mode,
                )   
            else:                
                loss, grads = train_batch_bptt(
                    model,
                    inputs,  # Full batch
                    targets,
                    init_state,
                    optimizer,
                    train_metrics,
                    cfg.training.grad_clip_val,
                    ignore_mask,
                    cfg.training.loss_aggregation,
                    cfg.training.prediction_mode
                )       
        grad_norm = jnp.sqrt(
            sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)])
        )
        model.eval()

        val_metrics.reset()
        for batch in val_dl:
            inputs, targets = batch

            ignore_mask = jnp.concatenate(
                [
                    jnp.zeros((inputs.shape[0], cfg.dataset.ignore_first_n_timesteps)),
                    jnp.ones_like(inputs)[:, cfg.dataset.ignore_first_n_timesteps :, 0],
                ],
                axis=1,
            )
            init_state = model.initialize_carry((inputs.shape[0], input_dim))

            if not cfg.training.eval_model_online:
                _ = eval_model(
                    model, inputs, ignore_mask, targets, init_state, val_metrics, cfg.training.loss_aggregation, cfg.training.prediction_mode
                )
            else:
                inputs = inputs.reshape(
                    inputs.shape[0],
                    jnp.maximum(total_seq_len // cfg.training.online_eval_subseq_len, 1),
                    cfg.training.online_eval_subseq_len,
                    input_dim,
                )
                ignore_mask = ignore_mask.reshape(
                    inputs.shape[0],
                    jnp.maximum(total_seq_len // cfg.training.online_eval_subseq_len, 1),
                    cfg.training.online_eval_subseq_len
                )
                init_online_eval = jnp.zeros((inputs.shape[0], targets.shape[-1]))
                _ = eval_model_online(
                    model, inputs, ignore_mask, targets, init_online_eval, init_state, val_metrics
                )

        print(
            f"Ep {epoch + 1}/{cfg.training.epochs}: Val Acc: {val_metrics.compute()['accuracy']:.4f}, Val Loss: {val_metrics.compute()['loss']:.4f}, Train Acc: {train_metrics.compute()['accuracy']:.4f}, Train Loss: {train_metrics.compute()['loss']:.4f}, Epoch Time: {time.time() - epoch_time:.2f}s, Grad Norm: {grad_norm:.4f}, Learning Rate: {optimizer.opt_state.hyperparams['learning_rate'].value:.4f}"
        )


        # Early stopping logic / working on this
        val_accuracy = val_metrics.compute()["accuracy"]
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            # Save best model parameters
            with open("best_model.pkl", "wb") as f:
                pickle.dump(nnx.split(model)[1], f)

            # print("New best model saved.")
            # print("Saving model to:", os.path.abspath("best_model.pkl"))

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Val Accuracy: {best_accuracy}")

    # Final evaluation on the test set
    best_state = pickle.load(open("best_model.pkl", "rb"))
    model_graph, _ = nnx.split(model)
    best_model = nnx.merge(model_graph, best_state)
    best_model.eval()

    test_metrics = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average())
    for batch in test_dl:
        inputs, targets = batch
        input_dim = inputs.shape[-1]
        ignore_mask = jnp.concatenate(
            [
                jnp.zeros((inputs.shape[0], cfg.dataset.ignore_first_n_timesteps)),
                jnp.ones_like(inputs)[:, cfg.dataset.ignore_first_n_timesteps:, 0],
            ],
            axis=1,
        )
        init_state = model.initialize_carry((inputs.shape[0], input_dim))
        _ = eval_model(
            best_model, inputs, ignore_mask, targets, init_state, test_metrics, cfg.training.loss_aggregation,
            cfg.training.prediction_mode
        )

    print(f"Final Test Loss: {test_metrics.compute()['loss']:.4f} - Final Test Acc: {test_metrics.compute()['accuracy'] * 100.0:.4f} %")

if __name__ == "__main__":
    main()
