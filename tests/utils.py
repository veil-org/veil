from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union
from typeguard import check_type

import veil
from veil.types import StringDict, StringList



class Autologgable(ABC):

    def __init__(self, 
        name:Optional[str] = None,
        log_params:Optional[StringList] = None,
        log_tags:StringDict = dict()
    ):
        self.name:Optional[str] = check_type(name, Optional[str])
        self.log_params:Optional[StringList] = check_type(log_params, Optional[StringList])
        self.log_tags:StringDict = check_type(log_tags, StringDict)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
        


class AutologgableInstance(Autologgable):
    
    def __call__(self, func:Callable, *args, **kwargs):

        class ProxyClass:

            @veil.run(
                name = self.name,
                log_params = self.log_params,
                log_tags = self.log_tags
            )
            def __call__(self, *args, **kwargs):
                return func(*args, **kwargs)
        
        proxy:ProxyClass = ProxyClass()
        return proxy(*args, **kwargs)



class AutologgableFunction(Autologgable):

    def __call__(self, func:Callable, *args, **kwargs):
        
        @veil.run(
            name = self.name,
            log_params = self.log_params,
            log_tags = self.log_tags
        )
        def proxy_function(*args, **kwargs):
            return func(*args, **kwargs)
        
        return proxy_function(*args, **kwargs)



class AutologgableCallSequence:

    def __init__(self):
        self.callables:List[Union[AutologgableFunction, AutologgableInstance]] = []

    def __call__(self, func:Callable, *args, **kwargs) -> List[Any]:
        result:List[Any] = []
        for callable in self.callables:
            result.append(callable(func, *args, **kwargs))
        return result
    
    def add(self, proxy:Union[AutologgableFunction, AutologgableInstance]) -> AutologgableCallSequence:
        self.callables.append(check_type(proxy, Union[AutologgableFunction, AutologgableInstance]))
    