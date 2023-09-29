from .dataset import Dataset

class Evaluate:
    def __init__(
            self, 
            dataset, 
            *ignore, 
            workload=[],
            run_len=2,
            timeout=None,
            max_retry=3,
        ):
        self.dataset = Dataset(dataset)
        self.workload = workload
        self.run_len = run_len
        self.timeout = timeout
        self.max_retry = max_retry