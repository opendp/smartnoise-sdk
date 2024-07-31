import chex
import jax.numpy as jnp
import jax


def get_update_weights_fn(num_arms, learning_rate=0.1):
    def update_weights(weights: chex.Array, reward: chex.Array):
        i = reward[0]
        p = weights[i] * 0.95 + 0.05 * ( 1/ num_arms)
        r = reward[1] / p
        current_weight = weights[i]
        weights = weights.at[i].set(current_weight * jnp.exp(learning_rate * r))
        weights_normed = weights / weights.sum()
        return weights_normed, None

    def update_weights_fn(weights, rewards):
        updated_weights = jax.lax.scan(update_weights, weights, rewards)[0]
        return updated_weights

    return update_weights_fn

def get_sample_arms_fn(num_arms, num_samples):
    def sample_arms(rng: chex.PRNGKey, weights: chex.Array):

        weights_normed_mix = weights * 0.95 + (1/num_arms) * 0.05
        a = jax.random.choice(rng, num_arms, replace=True, shape=(num_samples, ), p=weights_normed_mix)
        return a
    return sample_arms