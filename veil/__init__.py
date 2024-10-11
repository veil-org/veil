"""
TODO docstring.
"""
from __future__ import annotations

import mlflow
from typing import Optional
from veil.decorators import Autologger

from veil.types import StringDict, StringList

__version__ = "0.5.1"


#####################
# GLOBAL HELPERS!!!!
#####################

__global_autologger:Autologger = Autologger()

def set_autolog_enabled(enabled:bool) -> None:
    global __global_autologger
    __global_autologger.is_autolog_enabled = enabled



def is_autolog_enabled() -> bool:
    global __global_autologger
    return __global_autologger.is_autolog_enabled



def set_experiment_name(experiment_name:str) -> None:
    global __global_autologger
    __global_autologger.experiment_name = experiment_name



def get_experiment_name() -> str:
    global __global_autologger
    return __global_autologger.experiment_name



def set_tracking_uri(tracking_uri:str) -> None:
    global __global_autologger
    __global_autologger.tracking_uri = tracking_uri



def get_tracking_uri() -> str:
    global __global_autologger
    return __global_autologger.tracking_uri



def start_session(
    name:Optional[str] = None,
    log_tags:StringDict = dict()
):
    global __global_autologger
    return __global_autologger.start_session(
        name=name, 
        log_tags=log_tags
    )



def run(
    name: Optional[str] = None,
    log_params: Optional[StringList] = None,
    log_tags: StringDict = dict()
):
    global __global_autologger
    return __global_autologger.run(
        name = name,
        log_params = log_params,
        log_tags = log_tags
    )