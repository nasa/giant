"""
This module provides the :class:`UserOptionConfigured` mixin class that enables classes to be
configured using :class:`.UserOptions`-derived classes while maintaining the ability to reset
to the original configuration state.

The mixin pattern allows for clean separation of configuration logic and provides a consistent
interface for option management across different classes.

Example:
    Basic usage of the UserOptionConfigured mixin::

        from giant.utilities.options import UserOptions
        from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured
        from dataclasses import dataclass

        @dataclass
        class MyOptions(UserOptions):
            a: int = 5
            b: float = -32.1

        class MyUsefulClass(UserOptionConfigured[MyOptions], MyOptions):
            def __init__(self, options: MyOptions = None):
                super().__init__(MyOptions, options)

        # Usage
        my_useful_inst = MyUsefulClass()
        my_useful_inst.a = 6  # Make a change
        print(my_useful_inst.a)  # Output: 6
        my_useful_inst.reset_settings()  # Reset to original
        print(my_useful_inst.a)  # Output: 5

.. Note::
    The :class:`UserOptionConfigured` class should come first in the inheritance order
    due to Method Resolution Order (MRO) requirements.

"""

from typing import Generic, TypeVar

from giant.utilities.options import UserOptions


OptionsT = TypeVar("OptionsT", bound=UserOptions)
"""
Type variable bound to UserOptions for type safety
"""


class UserOptionConfigured(Generic[OptionsT]):
    """
    Mixin class providing UserOptions-based configuration with reset capability.
    
    This mixin class enables classes to be configured using :class:`UserOptions`-derived 
    classes and provides the ability to reset the class to its default (initially provided) 
    state.
    
    The class works in conjunction with the :class:`UserOptions` ABC to provide:
    
    * Automatic configuration of class instances based on provided options
    * Storage of original configuration for reset functionality  
    * Type-safe option handling through generic typing
    
    To use this mixin, subclass it with the :class:`UserOptions` subclass as the type 
    parameter::
    
        class MyUsefulClass(UserOptionConfigured[MyOptions], MyOptions):
            def __init__(self, options: MyOptions = None):
                super().__init__(MyOptions, options)
    
    :param OptionsT: The :class:`UserOptions`-derived class type for configuration
        
    :attr original_options: The original configuration used during initialization.
                            This is stored as a deep copy and used for reset operations.
    
    .. Warning::
        If options are not provided during initialization, default initialization of the
        options_type class will be used.
    """
    
    def __init__(self, options_type: type[OptionsT], *args, options: OptionsT | None = None, **kwargs) -> None: 
        """
        :param options_type: The type of the :class:`.UserOptions` to use
        :param options: An optional oinstance of `options_type` preconfigured.
        """
        
        super().__init__(*args, **kwargs)
        
        if options is None:
            options = options_type()
            
        options.apply_options(self)
            
        self._original_options: OptionsT = options
        """
        The original configuration for this class
        """
    
    def reset_settings(self) -> None:
        """
        Resets the class to the state it was originally initialized with.
        """
        
        self.original_options.apply_options(self)
        
    @property
    def original_options(self) -> OptionsT:
        """
        Get the original configuration options.
        
        :returns: OptionsT: The original options used during initialization.
            
        .. Warning::
            Modifying the returned object will affect reset behavior.
        """
        return self._original_options
    
    @original_options.setter
    def original_options(self, value: OptionsT) -> None:
        """
        Manually Set the original configuration options.
        
        :param value: The options to store as original configuration.
            
        .. note::
            Generally this shouldn't be used.  If you want to manually change options
            you can do so directly just by using 
            `my_options.apply_options(my_class_inst)`
        """
        self._original_options = value