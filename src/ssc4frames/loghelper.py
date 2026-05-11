#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: remstef
"""

# set up logging
import sys
import logging

__handler__ = [ ]
__logger__ = [ ]

__default_formatter__ = logging.Formatter('%(processName)s:%(threadName)s:%(asctime)s:%(name)s:%(levelname)s: %(message)s')

def add_handler(handler):
    __handler__.append(handler)
    return handler

def add_console_handler():
    consolehandler = logging.StreamHandler()
    consolehandler.setFormatter(__default_formatter__)
    return add_handler(consolehandler)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in __handler__:
        logger.addHandler(handler)
    __logger__.append(logger)
    return logger

add_console_handler()