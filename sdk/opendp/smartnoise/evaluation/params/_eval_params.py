class EvaluatorParams:
    """
	Defines the fields used to set evaluation parameters
    and consumed by the evaluator
	"""

    def __init__(
        self,
        repeat_count=500,
        numbins=0,
        binsize="auto",
        exact=False,
        alpha=0.05,
        bound=True,
        eval_first_key=False,
        significance_level=0.05,
    ):
        self.repeat_count = repeat_count
        self.numbins = numbins
        self.binsize = binsize
        self.exact = exact
        self.alpha = alpha
        self.bound = bound
        self.eval_first_key = eval_first_key
        self.sig_level = significance_level
