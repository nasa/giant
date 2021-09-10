# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Provides an iterator for generating unique random combinations from a population where order doesn't matter.

This is useful for performing RANSAC analysis as it ensures that the same sample sets are not chosen multiple times.
"""

from typing import Union

from itertools import combinations
from random import sample

try:
    from scipy.special import comb

except ImportError:
    from scipy.misc import comb


from .._typing import ARRAY_LIKE


class RandomCombinations:

    def __init__(self, population: Union[int, ARRAY_LIKE], combo_length: int, number_of_combos: int):
        """
        Iterate over ``number_of_combos`` random combinations of ``combo_length`` from ``population``.

        This iterator ensures unique combinations are returned.  If more combinations are requested than are possible
        then an exhaustive list is returned

        :param population: The population to choose from.  If specified as an integer then the population will be
                           range(int).
        :param combo_length: The length for each combination as an integer
        :param number_of_combos: the number of unique combinations you want as an integer
        """

        if isinstance(population, int):
            population = range(population)
            int_population = population
        else:
            int_population = range(len(population))

        self.population = population
        self.int_population = int_population

        self.combo_length = combo_length

        self.possible_combos = int(comb(len(self.population), self.combo_length))

        self.number_of_combos = number_of_combos

    def __iter__(self) -> tuple:

        if self.number_of_combos >= self.possible_combos:
            for combo in combinations(self.population, self.combo_length):

                yield combo
        else:
            used_samples = set()
            for _ in range(self.number_of_combos):
                new_sample = tuple(sorted(sample(self.int_population, self.combo_length)))
                while new_sample in used_samples:
                    new_sample = tuple(sorted(sample(self.int_population, self.combo_length)))

                used_samples.add(new_sample)

                yield tuple(self.population[ind] for ind in new_sample)
