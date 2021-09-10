# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Welcome to GIANT

This is a test
"""

from importlib.machinery import PathFinder
import warnings
import sys


warnings.filterwarnings("default", category=DeprecationWarning)


class __MyPathFinder(PathFinder):

    __deprecated = ['attitude']
    __fixed = ['rotations']

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):

        name = fullname

        for dep, fix in zip(cls.__deprecated, cls.__fixed):

            if dep in name:
                name = name.replace(dep, fix)

                warnings.warn('{} is deprecated. Please import {} instead'.format(dep, fix),
                              DeprecationWarning)

        res = super().find_spec(name, path=path, target=target)

        return res


sys.meta_path.append(__MyPathFinder())


class __DepWrapper:

    def __init__(self, wrapped, deprecated):

        self.wrapped = wrapped
        self.deprecated = deprecated

        self.__doc__ = self.wrapped.__doc__
        self.__name__ = self.wrapped.__name__
        self.__file__ = self.wrapped.__file__

    def __getattr__(self, item):

        if item in self.deprecated:
            warnings.warn('"{}" is deprecated. Please use "{}" instead'.format(item, self.deprecated[item]),
                          DeprecationWarning)
            item = self.deprecated[item]

        return getattr(self.wrapped, item)
