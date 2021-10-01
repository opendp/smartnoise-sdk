# https://github.com/hoangtranngoc/AirSim-RL/blob/master/gym_airsim/airsim_car_env.py
#  from configparser import ConfigParser
import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import copy
import math
import logging
from opendp.smartnoise.sql import PandasReader, PrivateReader
from opendp.smartnoise.evaluation._dp_verification import DPVerification
import opendp.smartnoise.evaluation._exploration as exp
from opendp.smartnoise.evaluation.learner._transformation import *
from opendp.smartnoise.evaluation.learner._computeactions import compute_action
from opendp.smartnoise.evaluation.evaluator._dp_evaluator import DPEvaluator
from opendp.smartnoise.evaluation.learner.util import create_simulated_dataset, generate_neighbors
from opendp.smartnoise.metadata.collection import *
from dp_singleton_query import DPSingletonQuery


class DPEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self,
        LearnerParams,
        PrivacyParams,
        EvaluatorParams,
        DatasetParams,
        querypool,
        available_actions,
    ):
        self.pp = PrivacyParams
        self.ev = EvaluatorParams
        self.dd = DatasetParams
        self.lp = LearnerParams
        self.state = 0
        self.query = None
        self.querypool = querypool
        self.available_actions = available_actions
        self.action_space = spaces.Discrete(int(len(self.available_actions)))
        self.observation_space = spaces.Discrete(int(self.lp.observation_space))  # number of query
        self.df, self.metadata = create_simulated_dataset(self.dd.dataset_size, "dataset")
        self.state_query_pair_base = {}
        self.pool = list(range(self.lp.observation_space - 2, 1, -1))
        self.info = {}
        self.output = []
        self.episode = 0
        self.reward = 0
        self.d1_dataset, self.d2_dataset, self.d1_metadata, self.d2_metadata = generate_neighbors(
            self.df, self.metadata
        )

    def QuerytoAST(self, query, meta, data):
        reader = PandasReader(meta, data)
        private_reader = PrivateReader(meta, reader, self.pp.epsilon)
        # query =  'SELECT Usage AS Usage, SUM(Usage) + 3 AS u FROM dataset.dataset GROUP BY Role'
        try:
            ast = private_reader.parse_query_string(query)
        except:
            return
        return ast

    def observe(self, query):
        """
        get ast of a query
        """
        return self.QuerytoAST(query, self.d1_metadata, self.df)

    def move(self, query, action):
        ast = self.observe(query)
        if ast is not None:
            new_query = self.available_actions[action]["method"](
                ast, self.available_actions[action]
            )
            return str(new_query)
        else:
            return "invalid"

    def step(self, action):
        original_query = self.state_query_pair_base[self.state]
        new_query = self.move(original_query, action)
        msg = {}
        msg["original_query"] = original_query
        msg["chosen_action"] = self.available_actions[action]["description"]
        msg["new_query"] = new_query
        if new_query == "invalid":
            msg["episode"] = self.episode
            msg["dpresult"] = "ActionnotValid_ASTnotAvailable"
            msg["reward"] = 0
            self.info = msg
            self.output.append(msg)
            return int(self.lp.observation_space) - 1, 0, True, self.info
        elif new_query == original_query:
            msg["episode"] = self.episode
            msg["dpresult"] = "ActionResultedSameQuery"
            msg["reward"] = 0
            self.info = msg
            self.output.append(msg)
            return self.state, 0, False, self.info
        else:
            if new_query in list(self.state_query_pair_base.values()):
                self.state = list(self.state_query_pair_base.keys())[
                    list(self.state_query_pair_base.values()).index(new_query)
                ]
            elif new_query not in list(self.state_query_pair_base.values()):
                if len(self.pool) > 1:
                    _ = self.pool.pop()
                    self.state_query_pair_base[_] = new_query
                    self.state = _
                else:
                    self.state == self.total_query - 1
                # compute reward
            dpresult, self.reward, message, d1, d2 = self._compute_reward(new_query)
            self.query = new_query
            # check if the episode is done
            done = self._isDone()
            # log info
            msg["episode"] = self.episode
            msg["dpresult"] = dpresult
            msg["reward"] = self.reward
            msg["message"] = message
            msg["d1"] = d1
            msg["d2"] = d2
            self.info = msg
            self.output.append(msg)
            return self.state, self.reward, done, self.info

    def reset(self):
        start = random.choice(self.querypool)
        self.query = start
        self.state = 0
        self.reward = 0
        self.state_query_pair_base = {0: start, self.lp.observation_space - 1: "invalid"}
        self.pool = list(range(self.lp.observation_space - 1, 0, -1))
        return 0

    def render(self, mode="human"):
        print(self.info)
        return  # nothing

    def _compute_reward(self, query):
        ast_transform = self.observe(query)
        d1_query = query
        d2_query = query.replace("d1.d1", "d2.d2")
        d1_dataset, d2_dataset, d1_metadata, d2_metadata = generate_neighbors(
            self.df, self.metadata
        )
        d1 = PandasReader(d1_metadata, d1_dataset)
        d2 = PandasReader(d2_metadata, d2_dataset)
        eval = DPEvaluator()
        pa = DPSingletonQuery()
        key_metrics = eval.evaluate(
            [d1_metadata, d1], [d2_metadata, d2], pa, query, self.pp, self.ev
        )
        message = None
        if key_metrics["__key__"].dp_res is None:
            dpresult = "DP_BUG"
            self.reward = 1
            message = key_metrics["__key__"].error
        elif key_metrics["__key__"].dp_res == False:
            self._game_ended = True
            dpresult = "DP_FAIL"
            self.reward = 20
            message = "dp_res_False"
        elif (
            key_metrics["__key__"].dp_res == True
            and key_metrics["__key__"].jensen_shannon_divergence == math.inf
        ):
            self._game_ended = True
            dpresult = "DP_BUG"
            self.reward = 20
            message = "jsdistance_is_inf"
        else:
            res_list = []
            for key, metrics in key_metrics.items():
                dp_res = metrics.dp_res
                js_res = metrics.jensen_shannon_divergence
                # ws_res = metrics.wasserstein_distance
                res_list.append([dp_res, js_res])
            dp_res = np.all(np.array([res[0] for res in res_list]))
            js_res = (np.array([res[1] for res in res_list])).max()
            # ws_res = (np.array([res[2] for res in res_list])).max()
            if dp_res == True:
                dpresult = "DP_PASS"
                self.reward = js_res
        return dpresult, self.reward, message, d1, d2

    def _isDone(self):
        # if query exceed the state_space
        if self.reward == 20:
            return True
        if self.state == int(self.lp.observation_space) - 1:
            return True
        if len(self.state_query_pair_base) >= int(self.lp.observation_space) - 1:
            return True
        return False
