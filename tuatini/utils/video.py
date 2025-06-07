import importlib
import logging


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning("'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder")
        return "pyav"
