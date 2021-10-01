from .dpgan import DPGAN
from .pategan import PATEGAN


# try:
from .dpctgan import DPCTGAN  # noqa
from .patectgan import PATECTGAN  # noqa

__all__ = ["DPCTGAN", "PATECTGAN", "DPGAN", "PATEGAN"]
# except ImportError as e:
#     import logging

#     logger = logging.getLogger(__name__)
#     logger.warning(
#         "Requires: pip install 'ctgan==0.2.2.dev1' for ctgan based synthesizers. Failed with exception {}".format(e))

#     __all__ = ["DPGAN", "PATEGAN"]
