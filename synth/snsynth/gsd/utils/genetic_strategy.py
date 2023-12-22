import jax.numpy as jnp
import jax
import chex
from flax import struct
from snsynth.gsd.utils import Dataset, Domain
from functools import partial
from typing import Tuple


@struct.dataclass
class EvoState:
    archive: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float64).max


@struct.dataclass
class PopulationState:
    # X: chex.Array
    row_id: chex.Array
    remove_row: chex.Array
    add_row: chex.Array


def get_best_fitness_member(
    x: chex.Array, fitness: chex.Array, state
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    best_in_gen = jnp.argmin(fitness)
    best_in_gen_fitness, best_in_gen_member = (
        fitness[best_in_gen],
        x[best_in_gen],
    )
    replace_best = best_in_gen_fitness < state.best_fitness
    best_fitness = jax.lax.select(
        replace_best, best_in_gen_fitness, state.best_fitness
    )
    best_member = jax.lax.select(
        replace_best, best_in_gen_member, state.best_member
    )
    return best_member, best_fitness, replace_best, best_in_gen


class SDStrategy:
    def __init__(self, domain: Domain,
                 data_size: int,
                 population_size: int = 50,
                 elite_size: int = 5,
                 debugging=False):

        self.population_size = population_size
        self.elite_size = elite_size
        self.data_size = data_size

        self.domain = domain
        self.num_devices = jax.device_count()
        self.domain = domain
        self.debugging = debugging
        self.update_candidate_vmap = jax.vmap(update_candidate, in_axes=(None, 0, 0))

    def initialize(
            self, rng: chex.PRNGKey
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = self.initialize_elite_population(rng)
        state = EvoState(
            archive=init_x.astype(jnp.float64),
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float64).max,
            best_member=init_x[0].astype(jnp.float64),
            best_fitness=jnp.finfo(jnp.float64).max
        )

        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_elite_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization

    @partial(jax.jit, static_argnums=(0,))
    def initialize_random_population(self, rng: chex.PRNGKey):
        pop = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng)
        return pop

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def update_elite_candidates(self,
            state: EvoState, # best candidate before updates
            population_delta: PopulationState,  # candidate deltas
            fitness: chex.Array):
        X = state.best_member
        idx = jnp.argsort(fitness)[0]
        fitness_elite = fitness[idx].reshape((1,))

        row_idx = population_delta.row_id[idx]
        remove_row = population_delta.remove_row[idx]
        add_row = population_delta.add_row[idx]
        X_elite_upt = update_candidate(X, row_idx, add_row).reshape((1, self.data_size, -1))
        return X_elite_upt, fitness_elite, remove_row, add_row

    @partial(jax.jit, static_argnums=(0,))
    def tell(
            self,
            x: chex.Array, # Candidates before updates
            fitness: chex.Array,
            state: EvoState,
    ) -> Tuple[EvoState, chex.Array, chex.Array]:
        """`tell` performance data for strategy state update."""
        state = self.tell_strategy(x, fitness, state)
        best_member, best_fitness, replace_best, best_in_gen = get_best_fitness_member(x, fitness, state)
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
        ), replace_best, best_in_gen

    def tell_strategy(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> EvoState:
        fitness_concat = jnp.concatenate([state.fitness, fitness])
        idx = jnp.argsort(fitness_concat)[0: self.elite_size]

        solution_concat = jnp.concatenate([state.archive, x])

        new_fitness = fitness_concat[idx]
        new_archive = solution_concat[idx]

        new_state = state.replace(
            fitness=new_fitness, archive=new_archive,
        )

        return new_state


####################################################################################################
####################################################################################################
####################################################################################################

class MutateStrategy(SDStrategy):
    def __init__(self, domain: Domain, data_size: int, population_size: int = 50, elite_size: int = 5, debugging=False):
        super().__init__(domain, data_size, population_size, elite_size, debugging)

        self.samplers = [jax.jit(self.domain.get_sampler(col, self.population_size)) for col in self.domain.attrs]
        self.column_ids = self.domain.sample_columns_based_on_logsize() # Mutate columns based on their cardinality
        self.sample_id = 0

        def muta_fn(
                X0,
                rng: chex.PRNGKey,
                mut_col: chex.Array,
                new_col_value: chex.Array
        ) -> PopulationState:
            n, d = X0.shape
            rem_row_idx = jax.random.randint(rng, minval=0, maxval=n, shape=(1,))
            removed_rows_muta = X0[rem_row_idx, :]
            added_rows_muta = removed_rows_muta.at[0, mut_col].set(new_col_value)
            pop_state = PopulationState(row_id=rem_row_idx, remove_row=removed_rows_muta, add_row=added_rows_muta)
            return pop_state
        self.muta_vmap = jax.jit(jax.vmap(muta_fn, in_axes=(None, 0, 0, 0)))

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        self.sample_id = (self.sample_id + 1) % len(self.column_ids)
        column = self.column_ids[self.sample_id]
        column_values = self.samplers[column](rng)
        pop = self.ask_strategy(rng, state, column, column_values)
        return pop


    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng_muta: chex.PRNGKey, state: EvoState, i: int, column_values: chex.Array):
        rng_muta, rng_mate, rng_swap = jax.random.split(rng_muta, 3)
        # Mutation
        column_id = (jnp.ones(shape=(self.population_size)) * i).astype(int)
        rng_muta_split = jax.random.split(rng_muta, self.population_size)
        pop_muta = self.muta_vmap(state.best_member, rng_muta_split, column_id, column_values)

        return pop_muta


class ContinuousDataStrategy(SDStrategy):
    def __init__(self, domain: Domain, data_size: int, population_size: int = 50, elite_size: int = 5, debugging=False):
        super().__init__(domain, data_size, population_size, elite_size, debugging)

        self.continuous_columns = domain.get_continuous_cols() + domain.get_ordinal_cols()
        self.num_cont_cols = len(self.continuous_columns)
        self.cont_col_indices = domain.get_attribute_indices(self.continuous_columns)
        self.column_sizes = [domain.size(col) for col in self.continuous_columns]
        self.cont_col_id = -1

        self.quantiles = jnp.linspace(0, 1, 32)

        def muta_fn(
                X0,
                rng: chex.PRNGKey,
                mut_col: chex.Array,
                column_quantiles: chex.Array
        ) -> PopulationState:
            rng1, rng2, rng3 = jax.random.split(rng, 3)
            n, d = X0.shape
            rem_row_idx = jax.random.randint(rng1, minval=0, maxval=n, shape=(1,))
            removed_rows_muta = X0[rem_row_idx, :]

            quantile_pos = jax.random.randint(rng2, minval=0, maxval=column_quantiles.shape[0]-1, shape=(1, ))[0]
            new_value_lower = column_quantiles[quantile_pos]
            new_value_upper = column_quantiles[quantile_pos+1]
            new_col_value = jax.random.uniform(rng3, minval=new_value_lower, maxval=new_value_upper)

            added_rows_muta = removed_rows_muta.at[0, mut_col].set(new_col_value)
            pop_state = PopulationState(row_id=rem_row_idx, remove_row=removed_rows_muta, add_row=added_rows_muta)
            return pop_state
        self.muta_vmap = jax.jit(jax.vmap(muta_fn, in_axes=(None, 0, None, None)))

    @partial(jax.jit, static_argnums=(0,))
    def compute_quantiles(self, values, column_size):
        thresholds = jnp.concatenate([jnp.zeros((1,)),
                                      jnp.quantile(values, q=self.quantiles),
                                      column_size * jnp.ones((1,))])
        return thresholds

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        self.cont_col_id = (self.cont_col_id + 1) % self.num_cont_cols
        column = self.cont_col_indices[self.cont_col_id]
        column_size = self.column_sizes[self.cont_col_id]
        pop = self.ask_strategy(rng, state, column, column_size)
        return pop


    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng_muta: chex.PRNGKey, state: EvoState, column_id: int, column_size: int):

        # Mutation
        thresholds = self.compute_quantiles(state.best_member[:, column_id], column_size)
        rng_muta_split = jax.random.split(rng_muta, self.population_size)
        pop_muta = self.muta_vmap(state.best_member, rng_muta_split, column_id, thresholds)

        return pop_muta



class SwapStrategy(SDStrategy):
    def __init__(self, domain: Domain, data_size: int, population_size: int = 50, elite_size: int = 5, debugging=False):
        super().__init__(domain, data_size, population_size, elite_size, debugging)

        self.num_colums = len(domain.attrs)
        self.column_count = -1
        def swap_fn(
                X0,
                rng: chex.PRNGKey,
                mut_col: chex.Array,
        ) -> PopulationState:

            n, d = X0.shape
            rem_row_idx = jax.random.randint(rng, minval=0, maxval=n, shape=(2,))
            removed_rows_muta = X0[rem_row_idx, :]
            value_1, value_2 = X0[rem_row_idx, mut_col]

            # Swap values
            added_rows_muta = removed_rows_muta.at[0, mut_col].set(value_2)
            added_rows_muta = added_rows_muta.at[1, mut_col].set(value_1)

            pop_state = PopulationState(row_id=rem_row_idx, remove_row=removed_rows_muta, add_row=added_rows_muta)
            return pop_state

        self.operator_vmap = jax.jit(jax.vmap(swap_fn, in_axes=(None, 0, 0)))

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        self.column_count = (self.column_count + 1) % self.num_colums
        pop = self.ask_strategy(rng, state, self.column_count)
        return pop

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, state: EvoState, i: int):
        column_id = (jnp.ones(shape=(self.population_size)) * i).astype(int)
        rng_muta_split = jax.random.split(rng, self.population_size)
        pop_muta = self.operator_vmap(state.best_member, rng_muta_split, column_id)

        return pop_muta



class CategoricalCrossoverStrategy(SDStrategy):
    def __init__(self, domain: Domain, data_size: int, population_size: int = 50, elite_size: int = 5, debugging=False,
                 column_updates=1):
        super().__init__(domain, data_size, population_size, elite_size, debugging)

        self.column_updates = column_updates
        self.num_colums = len(domain.attrs)
        self.column_count = 0
        cat_idx = jnp.array(domain.get_attribute_indices(domain.get_categorical_cols()).astype(int))

        def mate(
                X0, rng: chex.PRNGKey,
        ) -> PopulationState:
            n, d = X0.shape
            rng, rng1, rng2, rng3, rng_normal = jax.random.split(rng, 5)
            rem_row_idx = jax.random.randint(rng1, minval=0, maxval=n, shape=(1,))
            removed_rows_mate = X0[rem_row_idx, :]
            # Copy this row onto the dataset
            add_rows_idx = jax.random.randint(rng2, minval=0, maxval=n, shape=(1,))
            new_rows = X0[add_rows_idx, :]
            cat_col_pos = jax.random.randint(rng3, minval=0, maxval=cat_idx.shape[0], shape=(self.column_updates,))
            upt_col_idx = cat_idx[cat_col_pos]
            added_rows_mate = removed_rows_mate.at[0, upt_col_idx].set(new_rows[0, upt_col_idx])
            pop_state = PopulationState(row_id=rem_row_idx, remove_row=removed_rows_mate, add_row=added_rows_mate)
            return pop_state

        self.operator_vmap = jax.jit(jax.vmap(mate, in_axes=(None, 0)))

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        pop = self.ask_strategy(rng, state)
        return pop

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, state: EvoState):
        rng_split = jax.random.split(rng, self.population_size)
        pop = self.operator_vmap(state.best_member, rng_split)
        return pop




def update_candidate(X, row_id, add_row):
    X_new = X.at[row_id, :].set(add_row)
    return X_new


