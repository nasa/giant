import numpy as np

from typing import Self, Any


class AttributeEqualityComparison:
    """
    A base class that implements equality comparison based on attributes.

    This class provides methods to compare two objects of the same class
    based on their attributes. It handles comparison of numeric array-like
    attributes using numpy's allclose functionality.

    Usage:
        Inherit from this class to add attribute-based equality comparison
        to your custom classes.
        
        For example:
        
        .. code-block:: 
            class MyClass(AttributeEqualityComparison):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            obj1 = MyClass(1, [2, 3])
            obj2 = MyClass(1, [2, 3])
            obj3 = MyClass(1, [2, 4])

            print(obj1 == obj2)  # True
            print(obj1 == obj3)  # False
    """
    def __eq__(self, other: Any) -> bool:
        """
        Compare this object with another for equality by checking equality of all attributes.

        :param other: The object to compare with
        :return: True if the objects are equal, False otherwise
        """
        
        if not isinstance(other, self.__class__):
            return False
        
        if not set(self.__dict__.keys()) == set(other.__dict__.keys()):
            return False
        
        comp_dict = self.comparison_dictionary(other) 
        
        if all(comp_dict.values()):
            return True
        else:
            print(f"Not equal in some attributes: {comp_dict}")
            return False
    
    @staticmethod 
    def _value_comparison(val1: Any, val2: Any) -> bool:
        """
        Compare two values, handling array-like objects.
        
        This can be overriden if need be.

        :param val1: First value to compare
        :param val2: Second value to compare
        :return: True if values are equal, False otherwise
        """
        if isinstance(val1, (np.ndarray, list, tuple)) and isinstance(val2, (np.ndarray, list, tuple)):
            try:
                return np.allclose(val1, val2)
            except TypeError:
                # Fall back to regular equality for non-numeric arrays
                return bool(np.all(val1 == val2))
        return val1 == val2
        
    def comparison_dictionary(self, other: Self) -> dict[str, bool]:
        """
        Compares each attribute of self to other and stores the result in a dict mapping the attribute to the comparison result.
        
        Each attribute is compared with a call to the internal :meth:`._value_comparison` which by default uses numpy's allclose functionality
        for arraylike objects and standard equality checks for all others.  Therefore, if more detailed handling is needed, this 
        :meth:`._value_comparison` method can be overridden while maintaining the functionality in this method and the 
        class's equality comparison check.
        
        This assumes that other and self are the same type and have the same attributes.
        
        :param other: The other instance to compare with
        :return: A dictionary mapping attribute names to comparison results
        """

        return {
            key: self._value_comparison(getattr(self, key), getattr(other, key))
            for key in self.__dict__.keys()
        }