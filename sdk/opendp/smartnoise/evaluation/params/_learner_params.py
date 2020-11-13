class LearnerParams:
    """
    Defines the fields used to set halton space
    that is used for generating halton samples
    """
    def __init__(self):
        self.seedquery = ['SELECT SUM(Role) FROM d1.d1 GROUP BY Role']
        self.eps = 0.1
        self.lr = .8
        self.y = .9
        self.num_episodes = 200
        self.num_steps = 200
        self.observation_space = 1000
        self.columns = ['UserId', 'Role', 'Usage']
        self.numofquery = 1000
        self.MAXNODELEN=30
        
        

