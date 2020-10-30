#https://github.com/hoangtranngoc/AirSim-RL/blob/master/gym_airsim/airsim_car_env.py
#  from configparser import ConfigParser
import gym
from gym import spaces
import numpy as np
import pandas as pd
from opendp.smartnoise.sql import PandasReader, PrivateReader
from opendp.smartnoise.evaluation._dp_verification import DPVerification
import opendp.smartnoise.evaluation._exploration as exp
from opendp.smartnoise.evaluation.learner._transformation import *
from opendp.smartnoise.evaluation.learner._computeactions import compute_action
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams

import random
import copy
import logging
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
class DPEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, ep:LearnerParams, querypool, available_actions, dataset_size = 1000):
        self.dv = DPVerification(dataset_size=dataset_size)
        self.ep = ep
        self.state =  0
        self.query = ep.seedquery
        self.querypool = querypool
        self.available_actions = available_actions
        self.action_space = spaces.Discrete(int(len(self.available_actions)))
        self.observation_space = spaces.Discrete(int(ep.observation_space))# number of query
        self.total_query = ep.observation_space
        self.data, _, _, _ = self.dv.create_simulated_dataset()
        # self.data = pd.read_csv('simulation.csv')
        _, _, self.meta, _ = self.dv.generate_neighbors(load_csv=True)
        # self.meta = load_obj("simulation_meta_processed")
        self.state_query_pair_base = {0:ep.seedquery, ep.observation_space-1:"invalid"}
        self.pool = list(range(ep.observation_space-2, 1, -1))
        self.info = {}
        self.output=[]
        self.episode = 1
        self.reward = 0
       

  
    def QuerytoAST(self, query, meta, data):
        reader = PandasReader(meta, data)
        # prOptions = PrivateReaderOptions(censor_dims = False)
        # private_reader = PrivateReader(meta, reader, epsilon = 1, options=prOptions)
        private_reader = PrivateReader(meta, reader, self.epsilon)    
        try:
            ast = private_reader.parse_query_string(query) 
        except:
            return
        return ast

    def observe(self, query):
        """
        get ast of a query
        """
        return QuerytoAST(query, self.meta, self.data)

    def move(self, query, action):
        ast = self.observe(query)
        if ast is not None:
            new_query = self.available_actions[action]['method'](ast, self.available_actions[action])
            return str(new_query)
        else:
            return 'invalid'    


    def step(self, action):
        original_query = self.state_query_pair_base[self.state]
        new_query = self.move(original_query, action)
        msg={}
        msg['original_query']=original_query
        msg['chosen_action']=self.available_actions[action]['description']
        msg['new_query']=new_query
        if new_query == 'invalid' :
            msg['episode'] = self.episode
            msg['result']='ASTnotAvailable'
            msg['reward']=0
            self.info=msg
            self.output.append(msg)
            return self.total_query-1, 0, True,  self.info


        elif new_query == original_query:
            msg['episode'] = self.episode
            msg['result']='ActionnotValid'
            msg['reward']= 0
            result = "ActionnotValid"
            self.info=msg
            self.output.append(msg)
            return self.state, 0, False, self.info

        else:
            if new_query in list(self.state_query_pair_base.values()):
                self.state = list(self.state_query_pair_base.keys())[list(self.state_query_pair_base.values()).index(new_query)]
                self.reward, result, message, message_detail, d1, d2 = self._compute_reward(new_query)
            elif new_query not in list(self.state_query_pair_base.values()):
                if len(self.pool)>1:
                    _ = self.pool.pop()
                    self.state_query_pair_base[_] = new_query            
                    self.state = _   
                else:
                    self.state == self.total_query-1             
                # compute reward                     
                self.reward, result, message, message_detail, d1, d2 = self._compute_reward(new_query)
            self.query = new_query 
            # check if the episode is done
            done = self._isDone()
            print(done)
            # log info
            msg['episode'] = self.episode
            msg['result']= result
            msg['reward']=self.reward
            msg['message']=message
            msg['message_detail']=message_detail
            msg['d1'] = d1
            msg['d2'] = d2
            self.info=msg
            self.output.append(msg)
            return self.state, self.reward, done, self.info

    def reset(self):
        start = random.choice(self.querypool) 
        self.query = start
        self.state =  0
        self.reward = 0
        self.state_query_pair_base = {0:start, self.total_query-1:"invalid"}
        self.pool = list(range(self.total_query-2, 1, -1))
        return 0 

    def render(self, mode='human'):
        print(self.info)
        return #nothing


    def _compute_reward(self, query):
        #halton_sample = random.choice(self.halton_samples)
        ast_transform = self.observe(query)
        d1_query = query
        d2_query = query.replace("d1.d1", "d2.d2")  
        dp_res, ks_res, ws_res, d1, d2 = self.dv.dp_groupby_query_test_rl(d1_query, d2_query,repeat_count=500)
        message = None
        message_detail = None
        if dp_res is None and ks_res in ['ValueError_parsequerystring', 'ValueError_reader', 'exact_df_error']:
            result = "QUERY_INVALID"
            self.reward = 0
            message = ks_res
            if ws_res:
                message_detail = ws_res.replace("\n", "")
            
        elif dp_res is None and ks_res in ['num_cols_is_zero', 'ValueError_privatereader']: # syntax correct, but DP forced it not to be passed or not applicable to DP
            result = "QUERY_VALID_NOT_APPLICABLE_DP"
            self.reward = 1
            message = ks_res
            if ws_res:
                message_detail = ws_res.replace("\n", "")
            
        elif dp_res is None and ks_res in['noisy_df_empty', 'd1_d2 table empty', 'd1_d2 not merge with exact']: # syntax correct, but evaluator doesn't process such query
            result = "EVALUATOR_NO_RESULT"
            self.reward= 1
            message = ks_res
            if ws_res:
                message_detail = ws_res.replace("\n", "")

        elif dp_res is None and ks_res in['Error_inf_in_fD1']: # DP system bug
            result = "DP_BUG"
            self.reward= 20
            message = ks_res
            if ws_res:
                message_detail = ws_res.replace("\n", "")
            
        elif dp_res:# if query valid and dp test pass
            result = "DP_PASS"
            self.reward= ws_res 
            message = ks_res

        elif dp_res == False:
            self._game_ended = True
            result = "DP_FAIL"
            self.reward = 20
            message = ks_res
        return self.reward, result, message, message_detail, d1, d2

    def _isDone(self):
        # if query exceed the state_space
        if self.reward == 20:
            return True
        if self.state == self.total_query-1:
            return True
        if len(self.state_query_pair_base) >= self.total_query-1:
            return True
        return False

