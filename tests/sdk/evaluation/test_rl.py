import numpy as np
import pandas as pd 
import csv
import logging
test_logger = logging.getLogger("test-logger")
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.learner._dp_env import DPEnv
from opendp.smartnoise.evaluation.learner._generate import Grammar
from opendp.smartnoise.evaluation.learner._computeactions import compute_action
from opendp.smartnoise.evaluation.learner.util import write_to_csv
 


class TestQlearning():
    def TestQlearning(self, ep: LearnerParams):
        available_actions = compute_action(ep)
        querypool = ["SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"]
        env = DPEnv(ep, querypool, available_actions)
        # Set learning parameters
        eps = ep.eps
        lr = ep.lr
        y = ep.y
        num_episodes = 1
        num_steps = 2
        #Initialize table with all zeros
        Q = np.zeros([env.observation_space.n,env.action_space.n])
        for i in range(num_episodes):            
            logging.debug("%s episode" %i)
            #Reset environment and get first new observation
            s = env.reset()
            env.episode = i
            d = False
            j = 0
            while j < num_steps:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                if np.random.random() < eps: # explore
                    a = np.random.randint(env.action_space.n)
                else:
                    a = np.argmax((Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))) 
                #Get new state and reward from environment
                s1,r,d,info = env.step(a)
                #Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
                s = s1
                if d == True:
                    break
                assert((info['dpresult'] == 'DP_PASS') | (info['dpresult'] == 'ActionResultedSameQuery') | (info['dpresult'] == 'DP_BUG'))
            print('done')


Q = TestQlearning()
ep = LearnerParams()
Q.TestQlearning(ep)