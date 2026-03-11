from models.adlif_jax import AdLIF
from models.alif_jax import ALIF
from models.li_jax import LI
from models.brf_jax import BRF
import optax

def get_model_class(class_identifyer):
    print(f"Using model class: {class_identifyer}")
    if class_identifyer == "AdLIF":
        return AdLIF
    elif class_identifyer == "LI":
        return LI
    elif class_identifyer == "BRF":
        return BRF
    elif class_identifyer == "ALIF":
        return ALIF
    else:
        raise ValueError(f"Unknown model class: {class_identifyer}")
    
def lr_scheduler_factory(base_lr, steps_per_epoch, sch_cfg):
    print(f"Using learning rate scheduler: {sch_cfg}")
    if sch_cfg["name"] == "linear":
        return optax.linear_schedule(init_value=base_lr * sch_cfg["init_factor"], end_value=base_lr * sch_cfg["end_factor"], transition_steps=steps_per_epoch * sch_cfg["transition_epochs"])
    elif sch_cfg["name"] == "constant":
        return optax.constant_schedule(base_lr)
    elif sch_cfg["name"] == "cosine":
        return optax.cosine_decay_schedule(init_value=base_lr * sch_cfg["init_factor"], decay_steps=steps_per_epoch * sch_cfg["transition_epochs"], alpha=0.0, exponent=1.0)
    else:
        raise ValueError(f"Unknown learning rate scheduler: {sch_cfg['name']}")

def optimizer_factory(optim_cfg, learning_rate):
    print(f"Using optimizer: {optim_cfg['name']}")
    print(f"Learning rate: {learning_rate}")
    if optim_cfg["name"] == "RMSProp":
        return optax.inject_hyperparams(optax.rmsprop)(learning_rate=learning_rate)
    elif optim_cfg["name"] == "Adam":
        return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optim_cfg['name']}")
    

