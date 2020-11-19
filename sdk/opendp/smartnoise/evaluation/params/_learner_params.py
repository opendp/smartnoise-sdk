class LearnerParams:
    """
    Defines the fields used to set halton space
    that is used for generating halton samples
    """
    def __init__(self, eps=0.1, lr=0.8, y=0.9, num_episodes=200, num_steps=100, observation_space=1000, columns=['UserId', 'Role', 'Usage'], MAXNODELEN=30):
        self.eps = eps
        self.lr = lr
        self.y = y
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.observation_space = observation_space
        self.columns = columns
        self.MAXNODELEN= MAXNODELEN
        
        

