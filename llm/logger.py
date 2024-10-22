"""
Main logging module of the project.
"""
import logging
import os

class Logger(logging.Logger):

    log_dirpath = "./models/logs"
    log_dirpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), log_dirpath)
    )
    if not os.path.exists(log_dirpath):
        os.mkdir(path=log_dirpath)

    def __init__(self, filename: str, logger_name: str, mode: str = "w"):
        """
        Initialize the logger with the name.

        Args:
            filename (str) : The name of the log file.
            logger_name (str) : The name of the logger.
            mode (str) : The mode of the file to open.
        """
        super().__init__(name=logger_name)
        ffmt = logging.Formatter(
            fmt="(%(name)s)[%(levelname)-8s]: %(asctime)s - %(message)s ; module `%(module)s` in line %(lineno)s"
        )
        fhdlr = logging.FileHandler(
            filename=os.path.join(self.log_dirpath, filename + ".log"),
            mode=mode,
            encoding="utf-8"
        )
        fhdlr.setFormatter(fmt=ffmt)
        fhdlr.setLevel(level=logging.DEBUG)
        self.addHandler(hdlr=fhdlr)


# loggers
model_logger = Logger(filename="llm.model", logger_name="llm_logger")