import jax.numpy as jnp
import jax

def surrogate_gradient_FGI(z_raw, surr_id):
        if surr_id == "double_gaussian":
            return spike_double_gaussian_FGI(z_raw)
        elif surr_id == "straight_through":
            return straight_through_FGI(z_raw)
        elif surr_id == "slayer_wide":
            return spike_slayer_FGI(z_raw, 1, 0.4)
        elif surr_id == "slayer":
            return spike_slayer_FGI(z_raw, 5, 0.4)
        else:
            raise ValueError(f"Invalid surrogate gradient type {surr_id}")

def inject_gradient(z, x, d_z_d_x):
    """
    Inject gradient into the spike signal.
    Args:
        z: Spike signal
        x: Input signal
        d_z_d_x: Surrogate gradient
    Returns:
        Injected gradient
    """
    return jax.lax.stop_gradient(z) + (x - jax.lax.stop_gradient(x)) * jax.lax.stop_gradient(d_z_d_x)


def straight_through_FGI(x):
    """
    Straight-through gradient for binary spikes.
    Args:
        x: Input signal
    Returns:
        z: Spike signal
        d_z_d_x: Surrogate gradient
    """
    z = jnp.heaviside(x, 0.0)
    d_z_d_x = jax.lax.stop_gradient(jnp.ones_like(z))
    z = inject_gradient(z, x, d_z_d_x)
    return z, d_z_d_x

def spike_slayer_FGI(x, alpha, scale):
    # For example, use a surrogate spike (Heaviside) with an exponential surrogate gradient.
    z = jnp.heaviside(x, 0.0)
    d_z_d_x = scale * alpha / (2 * jnp.exp(alpha * jnp.abs(x)))
    z = inject_gradient(z, x, d_z_d_x)
    return z, jax.lax.stop_gradient(d_z_d_x)


def spike_gaussian_FGI(x):
    z = jnp.heaviside(x, 0.0)
    d_z_d_x = jax.scipy.stats.norm.pdf(x)
    z = inject_gradient(z, x, d_z_d_x)
    return z, jax.lax.stop_gradient(d_z_d_x)


# How they use it in the BHRF
def spike_double_gaussian_FGI(x):
    p = 0.15
    scale = 6.0
    len = 0.5

    sigma1 = len
    sigma2 = scale * len

    gamma = 0.5
    d_z_d_x = ( (1.0 + p) * gaussian(x, mu=0.0, sigma=sigma1) - 2.0 * p * gaussian(
        x, mu=0.0, sigma=sigma2
    ) ) * gamma
    z = jnp.heaviside(x, 0.0)
    z = inject_gradient(z, x, d_z_d_x)
    return z, jax.lax.stop_gradient(d_z_d_x)

# How they use it in the ALIF
def spike_double_gaussian_FGI_ALIF_ORIG(x):
    scale = 6.0
    lens = 0.5
    gamma = 0.5
    scale = 6.0
    hight = 0.15
    d_z_d_x = gaussian(x, mu=0.0, sigma=lens) * (1. + hight) \
            - gaussian(x, mu=lens, sigma=scale * lens) * hight \
            - gaussian(x, mu=-lens, sigma=scale * lens) * hight
    d_z_d_x = d_z_d_x * gamma
    z = jnp.heaviside(x, 0.0)
    z = inject_gradient(z, x, d_z_d_x)
    return z, jax.lax.stop_gradient(d_z_d_x)


def gaussian(x, mu: float = 0.0, sigma: float = 1.0):
    return (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
        -((x - mu) ** 2) / (2.0 * (sigma**2))
    )


def plot_surrogate_derivative(
    z,
    d_z_d_x,
    surrogate_gradient,
    title="Surrogate Gradient",
    xlabel="x",
    ylabel="Surrogate Gradient",
):
    import matplotlib.pyplot as plt

    plt.plot(z, d_z_d_x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 1000)
    z, d_z_d_x = spike_double_gaussian_FGI(x)
    plot_surrogate_derivative(
        x, d_z_d_x, "Double Gaussian Surrogate Gradient", "Double Gaussian"
    )
    x = np.linspace(-5, 5, 1000)
    z, d_z_d_x = spike_double_gaussian_FGI_ALIF_ORIG(x)
    plot_surrogate_derivative(
        x, d_z_d_x, "Double Gaussian Surrogate Gradient ALIF", "ALIF Double Gaussian"
    )
    x = np.linspace(-5, 5, 1000)
    z, d_z_d_x = spike_gaussian_FGI(x)
    plot_surrogate_derivative(
        x, d_z_d_x, "Gaussian Surrogate Gradient", "Gaussian"
    )
    x = np.linspace(-5, 5, 1000)
    z, d_z_d_x = spike_slayer_FGI(x, 5, 0.4)
    plot_surrogate_derivative(
        x, d_z_d_x, "Slayer Surrogate Gradient", "Slayer"
    )
    x = np.linspace(-5, 5, 1000)