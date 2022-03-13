"""RDT Exceptions."""


class NotFittedError(Exception):
    """Error to raise when ``transform`` or ``reverse_transform`` are used before fitting."""