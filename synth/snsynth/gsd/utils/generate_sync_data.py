import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
import time
from snsynth.gsd.utils import Dataset, Domain
from snsynth.gsd.utils.genetic_strategy import (PopulationState, MutateStrategy, SwapStrategy,
                                                CategoricalCrossoverStrategy, ContinuousDataStrategy)
from snsynth.gsd.utils.mw_jit import get_sample_arms_fn, get_update_weights_fn

AVAILABLE_GENETIC_OPERATORS = ['mutate', 'continuous', 'cross', 'swap']

def generate(key,
             domain: Domain,
             N_prime: int,
             num_generations: int,
             private_statistics: jnp.ndarray,
             statistics_fn: Callable,
             genetic_operators=('mutate', ),
             public_dataset: Dataset = None,
             early_stop_threshold: float = 0.0001,
             print_progress=False) -> Dataset:

    strategies = [
    ]
    if 'mutate' in genetic_operators:
        strategies.append(MutateStrategy(domain=domain, data_size=N_prime))
    if 'continuous' in genetic_operators:
        if len(domain.get_continuous_cols() + domain.get_ordinal_cols()) > 0:
            strategies.append(ContinuousDataStrategy(domain=domain, data_size=N_prime))
    if 'swap' in genetic_operators:
        strategies.append(SwapStrategy(domain=domain, data_size=N_prime))
    if 'cross' in genetic_operators:
        if len(domain.get_categorical_cols()) > 0:
            strategies.append(CategoricalCrossoverStrategy(domain=domain, data_size=N_prime, column_updates=1))
        if len(domain.get_categorical_cols()) > 1:
            strategies.append(CategoricalCrossoverStrategy(domain=domain, data_size=N_prime, column_updates=2))

    if public_dataset is not None: assert N_prime == len(public_dataset.df), "Public data must have size=N_prime"

    private_statistics = jnp.array(private_statistics)

    early_stop = max(100, N_prime)
    def fitness_fn(X):
        fitness = jnp.linalg.norm(private_statistics - statistics_fn(X), ord=2) ** 2
        return fitness
    fitness_fn_vmap = jax.vmap(fitness_fn, in_axes=(0, ))

    def update_fitness_fn(stats: jnp.ndarray, pop_state: PopulationState):
        # 1) Update the statistics of this synthetic dataset
        rem_row = pop_state.remove_row
        add_row = pop_state.add_row
        num_rows = rem_row.shape[0]
        add_stats = (num_rows * statistics_fn(add_row))
        rem_stats = (num_rows * statistics_fn(rem_row))
        upt_sync_stat = stats.reshape(-1) + add_stats - rem_stats
        fitness = jnp.linalg.norm(private_statistics - upt_sync_stat / N_prime, ord=2) ** 2
        return fitness
    update_fitness_fn_vmap = jax.vmap(update_fitness_fn, in_axes=(None, 0))
    update_fitness_fn_jit = jax.jit(update_fitness_fn_vmap)

    # INITIALIZE STATE
    key, subkey = jax.random.split(key, 2)

    state = strategies[0].initialize(subkey)

    if public_dataset is not None:
        init_sync = public_dataset.to_numpy()
        temp = init_sync.reshape((1, init_sync.shape[0], init_sync.shape[1]))
        new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
        state = state.replace(archive=new_archive)

    # Compute fitness of initial candidates
    elite_fitness = fitness_fn_vmap(state.archive)

    best_member_id = elite_fitness.argmin()
    state = state.replace(
        fitness=elite_fitness,
        best_member=state.archive[best_member_id],
        best_fitness=elite_fitness[best_member_id]
    )

    elite_stat = N_prime * statistics_fn(state.best_member)  # Statistics of best SD

    def update_elite_stat(elite_stat_arg,
                          replace_best,
                          remove_row,
                          add_row,
                          ):
        num_rows = remove_row.shape[0]

        new_elite_stat = jax.lax.select(
            replace_best,
            elite_stat_arg
            - (num_rows * statistics_fn(remove_row))
            + (num_rows * statistics_fn(add_row)),
            elite_stat_arg
        )
        return new_elite_stat
    update_elite_stat_jit = jax.jit(update_elite_stat)

    LAST_LAG_FITNESS = 1e9


    # Set up strategy samplers
    num_strategies = len(strategies)
    update_weight = jax.jit(get_update_weights_fn(num_strategies))
    sample = jax.jit(get_sample_arms_fn(num_strategies, early_stop+1))
    strategy_rewards = []
    key, key_sample = jax.random.split(key, 2)
    weights = jnp.ones(num_strategies) / num_strategies
    strategy_ids = list(np.array(sample(key_sample, weights)))

    t0 = time.time()
    keys = jax.random.split(key, num_generations)
    for t, ask_subkey in enumerate(keys):
        strategy_id = strategy_ids.pop()
        g_strategy = strategies[strategy_id]

        # ASK: Produce a set of new synthetic data candidates
        population_state = g_strategy.ask(ask_subkey, state)

        # FIT: Compute score of each synthetic dataset candidate
        fitness = update_fitness_fn_jit(elite_stat, population_state)

        # Update the best new candidates
        best_new_candidates, best_candidate_fitness, remove_row, add_row = g_strategy.update_elite_candidates(state, population_state, fitness)

        # TELL: Replace the best candidate
        state, rep_best, best_id = g_strategy.tell(best_new_candidates, best_candidate_fitness, state)

        # UPDATE: Update the statistics of the best synthetic datasets so far
        elite_stat = update_elite_stat_jit(elite_stat, rep_best, remove_row, add_row).block_until_ready()

        strategy_rewards.append([strategy_id, rep_best])

        if t % early_stop == 0:
            print_progress_fn(t, state.best_fitness, LAST_LAG_FITNESS, weights, time.time() - t0, print_progress)

        # Early Stop:
        if t % early_stop == 0 and t > 0:

            key, key_sample = jax.random.split(key, 2)
            rewards_jnp = jnp.array(strategy_rewards)
            weights = update_weight(weights, rewards_jnp)
            strategy_ids = list(np.array(sample(key_sample, weights)))
            strategy_rewards = []           # reset strategy count

            if check_early_stop(t, state.best_fitness, LAST_LAG_FITNESS, N_prime, early_stop_threshold, print_progress): break
            LAST_LAG_FITNESS = state.best_fitness

    sync_dataset = Dataset.from_numpy_to_dataset(domain, state.best_member)
    return sync_dataset

def check_early_stop(t, best_fitness, LAST_LAG_FITNESS, stop_early_min_generation, early_stop_threshold, print_progress: False):
    if (t % stop_early_min_generation) > 0: return False
    if (t <= stop_early_min_generation): return False
    if t == 0: return False
    loss_change = jnp.abs(LAST_LAG_FITNESS - best_fitness) / LAST_LAG_FITNESS
    if loss_change < early_stop_threshold:
        if print_progress:
            print(f'\t\t ### Stop early at {t} ###')
        return True
    return False

def print_progress_fn(t, best_fitness, LAST_LAG_FITNESS,  strategy_profile: jnp.ndarray, time, print_progress: False):
    if print_progress:
        loss_change = jnp.abs(LAST_LAG_FITNESS - best_fitness) / LAST_LAG_FITNESS
        print(f'Gen={t:>10}: '
              f'fitness={best_fitness:>10.9}. '
              f'fitness change={loss_change:.8f}',
              f' Strategy weights: ',
              strategy_profile.round(3),
              f'time={time:<5.2f}(s)')


