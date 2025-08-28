import warnings

from dataclasses import dataclass, fields

from typing import Optional, Callable, Union, Tuple, Dict, TypeVar, Any

from abc import ABCMeta, abstractmethod

@dataclass
class UserOptions(metaclass=ABCMeta):
    """
    This is an abstract class used to create a dataclass of user options. 
    
    These options are used to set defaults for parameters set inside the associated
    class for the options. 
    
    Example:
        StarIDOptions contains the default options for the StarID class. 
        
    Custom objects built from this abstract class must follow the naming scheme <callable_name>Options and be loaded into the
    options keyword argument for callable_name.__init__(). 
    
    To apply options to your class, the UserOptions.apply_options() method should be invoked. 
    
    for example:
        >>> @dataclass
        >>> class ExampleOptions(UserOptions):
        >>>     example_var : int = 1234
            
        >>> class Example(UserOptions):
        >>>     def __init__(self, positional_argument, options = None):
        >>>         if options is None:
        >>>             options = ExampleOptions()
        >>>         options.apply_options(self) #apply the options as attributes of self
        >>> my_example = Example()
        >>> print(my_example.example_var)
        ...     1234
    """
    
    def override_options(self):
        '''
        This method is used for special cases when certain options should be overwritten
        '''
        pass
    
    def local_vars(self, variables : list = [], target : Optional[Callable] = None) -> Union[list, TypeVar]:
        '''
        this is used to overwrite options in variables only used in __init__
        '''
        locs =  [getattr(self, str(var)) for var in variables]
        
        # clear variables that are local
        self._clear_globals(variables=variables)
        if target is not None:
            self._clear_object_locals(variables=variables, target=target)
            
        #return the variables so the user can pull them to local target
        if len(locs) == 1:
            return locs[0]
        else:
            return locs
    
    def _clear_globals(self, variables = []):
        """
        Remove parameters from this target
        """
        for var in variables:
            try:
                self.__dict__.pop(str(var))
                self.__annotations__.pop(str(var))
            except:
                pass
        
    def _clear_object_locals(self, variables:list=[], target:Optional[Callable]=None):
        """
        Remove parameters from loaded target
        """
        for var in variables:
            try:
                target.__dict__.pop(str(var))
            except:
                pass
            
    def apply_options(self, target: object) -> None:
        """
        Update the options as attributes of the object class
        
        :param target: the instance that we are to update
        """
        target.__dict__.update(self.options_dict)
    
    @property
    def options_dict(self) -> Dict:
        """
        Determine the options input to the dataclass.
        
        This property method will ignore all internal properties and functions
        """
        
        self.override_options()
        self._options = {key: self.__dict__[key] for key in list(self.__annotations__)}
        return self._options
