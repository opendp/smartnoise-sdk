import numpy as np
import pandas as pd
import csv
import logging
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.params._dataset_params import DatasetParams
from opendp.smartnoise.evaluation.learner._dp_env import DPEnv
from opendp.smartnoise.evaluation.learner._generate import Grammar
from opendp.smartnoise.evaluation.learner._computeactions import compute_action
from opendp.smartnoise.evaluation.learner.util import (
    create_simulated_dataset,
    generate_neighbors,
    generate_query,
    write_to_csv,
)

logging.basicConfig(filename="Q-learning.log", level=logging.DEBUG)


class QLearning:
    """
    Use Q-learning to conduct reinforcement learning based query search in evaluator
    """

    def __init__(self, learner_params):
        self.lp = learner_params
        self.pp = PrivacyParams(epsilon=1.0)
        self.ev = EvaluatorParams(repeat_count=100)
        self.dd = DatasetParams(dataset_size=500)

    def learn(self, query_pool, export_as_csv=False):
        # available transformation actions to AST
        available_actions = compute_action(self.lp)
        env = DPEnv(self.lp, self.pp, self.ev, self.dd, query_pool, available_actions)
        # Set learning parameters
        eps = self.lp.eps
        lr = self.lp.lr
        y = self.lp.y
        num_episodes = self.lp.num_episodes
        num_steps = self.lp.num_steps
        # Initialize table with all zeros
        Q = np.zeros([env.observation_space.n, env.action_space.n])
        for i in range(num_episodes):
            logging.debug("%s episode" % i)
            # Reset environment and get first new observation
            s = env.reset()
            env.episode = i
            logging.debug("%s available actions" % len(env.available_actions))
            d = False
            j = 0
            while j < num_steps:
                j += 1
                print("step %s" % j)
                print("original state:", env.state)
                # Choose an action by greedily (with noise) picking from Q table
                if np.random.random() < eps:  # explore
                    a = np.random.randint(env.action_space.n)
                else:
                    a = np.argmax(
                        (Q[s, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i + 1)))
                    )
                # Get new state and reward from environment
                s1, r, d, info = env.step(a)
                logging.debug("%s step" % j)
                logging.debug(info)
                # Update Q-Table with new knowledge
                Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
                s = s1
                env.render()
                if d == True:
                    break
        if export_as_csv:
            write_to_csv("Q-learning.csv", env.output, flag="qlearning")
        else:
            return env.output
