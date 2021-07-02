from typing import List

class Privacy:
    def __init__(self, *ignore, epsilon:float=1.0, delta:float=10E-16, alphas:List[float]=[], neighboring:str="addremove"):
        """Privacy parameters.  Values are keyword-only.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.alphas = alphas
        if neighboring not in ["addremove", "substitute"]:
            raise ValueError("Neighboring definition must be 'addremove' or 'substitute'")
        self.neighboring = neighboring