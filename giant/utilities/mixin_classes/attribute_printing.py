"""
This module provides a class implementing default __str__ and __repr__ functionality.
"""
class AttributePrinting: 
    
    """
    A mixin class that provides __str__ and __repr__ functionality.

    This mixin implements __str__ and __repr__ methods which print the class
    signature and attributes for any subclass using standard attributes.

    For any attributes which start with an underscore, it checks if there is a
    corresponding property not starting with an underscore and reports that instead.
    """
    
    def _build_representation(self, attribute_repr: bool) -> str:
        """
        Implements the basic functionality of turning the class into a string including all attributes.
        
        :param attribute_repr: Whether to call repr on attributes instead of str.
        """
        
        class_name = self.__class__.__name__
        attributes = []
        for attr, value in self.__dict__.items():
            if attr.startswith('_'):
                prop_name = attr.lstrip('_')
                if hasattr(self.__class__, prop_name) and isinstance(getattr(self.__class__, prop_name), property):
                    attr = prop_name
                    value = getattr(self, prop_name)
            if attribute_repr:
                attributes.append(f"{attr}={value!r}".replace('\n', ''))
            else:
                attributes.append(f"{attr}={value}".replace('\n', ''))
        return f"{class_name}({', '.join(attributes)})"

    def __str__(self) -> str:
        """
        Returns a string representation of the object.

        :return: A string containing the class name and its attributes.
        """
        return self._build_representation(False)

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the object.

        :return: A string containing the class name and its attributes.
        """
        return self._build_representation(True)
