import numpy as np
import pandas as pd 
import csv
import logging

from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.learner._dp_env import DPEnv
from opendp.smartnoise.evaluation.learner._generate import Grammar
from opendp.smartnoise.evaluation.learner._computeactions import compute_action
from opendp.smartnoise.evaluation.learner.util import write_to_csv
logging.basicConfig(filename="Q-learning.log", level=logging.DEBUG)

class Qlearning():
    """
    Use Q-learning to conduct reinforcement learning based query search in evaluator
    """
   
    def Qlearning(self, ep: LearnerParams):       
        generate query pool
        with open ("select.cfg", "r") as cfg:
            rules=cfg.readlines()
            grammar = Grammar(ep)
            numofquery = grammar.numofquery
            grammar.load(rules)


        text_file = open("querypool.txt", "w")
        querypool = [] 
        for i in range(numofquery):   
            text_file.write(str(grammar.generate('statement')))
            text_file.write('\n')
            querypool.append(str(grammar.generate('statement')))
        text_file.close()

        fin1 = open("QueryPool.txt", "rt")
        querypool = [] 
        toreturn = []
        for line in fin1:
            querypool.append(line[:-1])

        # available transformation actions to AST
        available_actions = compute_action(ep)

        env = DPEnv(ep, querypool, available_actions)
        # Set learning parameters
        eps = ep.eps
        lr = ep.lr
        y = ep.y
        num_episodes = ep.num_episodes
        num_steps = ep.num_steps
        #Initialize table with all zeros
        Q = np.zeros([env.observation_space.n,env.action_space.n])
        for i in range(num_episodes):            
            logging.debug("%s episode" %i)
            #Reset environment and get first new observation
            s = env.reset()
            env.episode = i
            logging.debug("%s available actions" %len(env.available_actions))
            d = False
            j = 0
            while j < num_steps:
                j+=1
                print("step %s" %j)
                print("original state:", env.state)
                #Choose an action by greedily (with noise) picking from Q table
                if np.random.random() < eps: # explore
                    a = np.random.randint(env.action_space.n)
                else:
                    a = np.argmax((Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))) 
                #Get new state and reward from environment
                s1,r,d,info = env.step(a)
                logging.debug("%s step" %j)
                logging.debug(info)
                #Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
                s = s1
                env.render()
                if d == True:
                    break
            write_to_csv('Q-learning.csv', env.output, flag='qlearning')        


Q = Qlearning()
ep = LearnerParams()
Q.Qlearning(ep)
