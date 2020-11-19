## Reinforcement Learning Based Query Search for Differential Privacy Evaluator
#### This documentation explains how to use bandit.py and qlearning.py in learner to do query search step for DP evaluator.
#### The DP evaluator that assesses any differential privacy (DP) algorithm for compliance with the DP promises was released open source as part of Smartnoise. Certain variations of queries may reveal interesting privacy pitfalls. For example, “SELECT married, SUM(CASE WHEN pid=1 THEN 100000 ELSE 1 END) AS n FROM PUMS GROUP BY married”, this query may return a result like this:
| married        | n            |
| :------------- | :----------: | 
|  False         | 451  |
| True           | 100548 |

#### One may easily infer from the result that user with pid=1 is married. We seek to optimize the query search step using bandit and Q-learning approach. The goal is to help user to catch DP bugs(if any) for a DP test they built. 
#### 1. In the bandit approach, a number of queries will be generally according to a set of grammer rules, and will be tested in batch. A report will be generated for user to scan through the dp test results on large batch of auto generated queries.Exampler code:
    b = Bandit(PrivacyParams, EvaluatorParams, DatasetParams)
    querypool = generate_query(100)
    b.bandit(querypool)
##### - input: To initiate Bandit, you'll need PrivacyParams, EvaluatorParams and DatasetParams to be defined. Querypool will also be needed as a list of SQL queries. You can utilize generate_query,py, which does brute force SQL Generation via context free grammer for you.
##### - default: PrivacyParams(epsilon=1.0), EvaluatorParams(repeat_count=100),DatasetParams(dataset_size=500). 
##### - output: return a list of each query's DP test result (dict, contains 'dpresult', 'error', 'js_distance', 'query')

#### 2. In the Q-learning approach, a seed query is given randomly, and a random action towards query AST will be executed, a reward of 0 (invalid query), 1(valid query), jenson_shannon distance of probability distribution of repeated query response as reward if pass the DP test, 20 if DP test fail. The agent will learn to manipuate the query to be more complex until fail the DP test in order to get higher reward.  A report will be generated for user to scan through the dp test results. Exampler code:
    b = QLearning(LearnerParams, PrivacyParams, EvaluatorParams, DatasetParams)
    querypool = generate_query(1000)
    b.qlearning(querypool)
##### - input: To initial Qlearning, you'll need LeanerParams PrivacyParams, EvaluatorParams and DatasetParams to be defined.
##### - default: PrivacyParams(epsilon=1.0), EvaluatorParams(repeat_count=100),DatasetParams(dataset_size=500), LearnerParams(observation_space=1000, num_episodes=200, num_steps=100, eps = 0.1, lr = .8, y = .9, columns = ['UserId', 'Role', 'Usage'], MAXNODELEN=30). 
- observation_space: number of query uplimit in the state space
- eps: exploration epsilon
- lr: learning rate
- y: discount rate
- columns: columns in dataset 
- MAXNODELEN: max number of nodes in a query (to limit the length of the SQL query)
##### - output: return a list of each query's DP test result (dict, contains 'original_query', 'chosen_action', 'new_query', 'episode', 'dpresult', 'reward', 'message', 'd1', 'd2')




