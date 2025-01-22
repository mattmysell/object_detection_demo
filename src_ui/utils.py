#!/usr/bin/env python3
"""
Code for utility functions such as logging.
"""
# Standard Libraries
from contextvars import ContextVar
from logging import Formatter, INFO, StreamHandler, Filter, Logger

# Installed Libraries

# Local Files

CONTEXT_VALUE = ContextVar("context")

class ContextFilter(Filter): #pylint: disable=too-few-public-methods
    """
    Filter to get and add in the context value to logs.
    """
    def filter(self, record):
        """
        Filter to get and add in the context value to logs, it is okay if the value is not set yet.
        """
        record.context = CONTEXT_VALUE.get("")
        return True

class ContextLogger(Logger): #pylint: disable=too-few-public-methods
    """
    Logger class with ability to set correlation id.
    """
    def __init__(self, name, level=INFO):
        """
        Initialize as a streaming logger, with a context filter and format.
        """
        super().__init__(name, level)

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)
        self.addFilter(ContextFilter())

    def set_context(self, context_value: str):
        """
        Set the global context value.
        """
        CONTEXT_VALUE.set(context_value)
