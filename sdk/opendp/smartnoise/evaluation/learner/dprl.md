# This documentation explains how to use bandit and
#### The stochastic evaluator that assesses any differential privacy (DP) algorithm for compliance with the DP promises was released open source as part of Smartnoise. Certain variations of queries may reveal interesting privacy pitfalls. For example, “SELECT married, SUM(CASE WHEN pid=1 THEN 100000 ELSE 1 END) AS n FROM PUMS GROUP BY married”, this query may return a result like this:
| married     | n    |
| :------------- | :----------: | -----------: |
|  False | 451  
| True  | 100548 |
#### One may easily infer from the result that user with pid=1 is married. We seek to optimize the query search step using bandit and Q-learning approach. The goal is to help user to catch DP bugs(if any) for a DP test they built. 
#### In the bandit approach, a number of queries will be generally according to a set of grammer rules, and will be tested in batch. A report in csv format will be generated for user to scan through the dp test results on large batch of auto generated queries.Exampler code:
    b = bandit()
    ep = LearnerParams()
    b.generate_query(ep)
#### In the Q-learning approach, a seed query is given randomly, and a random action towards query AST will be executed, a reward of 0 (invalid query), 1(valid query), jenson_shannon distance of probability distribution of repeated query response as reward if pass the DP test, 20 if DP test fail. The agent will learn to manipuate the query to be more complex until fail the DP test in order to get higher reward.  A report in csv format will be generated for user to scan through the dp test results. Exampler code:
    Q = Qlearning()
    ep = LearnerParams()
    Q.Qlearning(ep)


