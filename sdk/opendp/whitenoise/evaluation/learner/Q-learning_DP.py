import numpy as np
import pandas as pd 
import csv
import logging

from opendp.whitenoise.evaluation.params._learner_params import LearnerParams
from opendp.whitenoise.evaluation.learner._dp_env import DPEnv
from opendp.whitenoise.evaluation.learner._generate import Grammar
from opendp.whitenoise.evaluation.learner._computeactions import compute_action

logging.basicConfig(filename="Q-learning.log", level=logging.DEBUG)

class Q_learning():
    """
    Use Q-learning to conduct reinforcement learning based query search in evaluator
    """
    def write_to_csv(self, filename, data):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = ['original_query', 'chosen_action', 'new_query', 'episode', 'result', 'reward', 'message', 'message_detail', 'd1', 'd2'], extrasaction='ignore')
            writer.writeheader()
            for i in data:
                writer.writerow(i)

    def Q_learning(self, ep: LearnerParams):       
        # generate query pool
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
                print("%s step" %j)
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
            self.write_to_csv('Q-learning.csv', env.output)        


# Q = Q_learning()
# ep = LearnerParams()
# Q.Q_learning(ep)
