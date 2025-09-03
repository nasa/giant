"""
Provides an iterator for generating unique random combinations from a population where order doesn't matter.

This is useful for performing RANSAC (Random Sample Consensus) analysis, a method for robust fitting of models
in the presence of outliers. It ensures that the same sample sets are not chosen multiple times, improving
the efficiency of the RANSAC algorithm.
"""

from typing import Union, Iterator, Sequence, Any

from itertools import combinations
from random import sample

from scipy.special import comb


from giant._typing import ARRAY_LIKE, BasicSequenceProtocol


class RandomCombinations:
    """
    Iterate over `number_of_combos` random combinations of `combo_length` from `population`.

    This iterator ensures unique combinations are returned. If more combinations are requested than are possible,
    then an exhaustive list is returned. This is particularly useful for RANSAC analysis, where we need to
    generate multiple unique subsets of data points to fit models.
    
    The population that combinations are coming from can either be provided directly, or an integer can be provided to 
    create a range based population.
    
    For example:
    
        >>> from giant.utilities.random_combination import RandomCombination
        >>> # works for an integer sequence
        >>> for sample in RandomCombination(10, 2, 3): print(sample)
        (2, 6)
        (1, 7)
        (1, 8)
        >>> # works for any sequence like
        >>> for sample in RandomCombination('abcdefghijk', 2, 3): print(sample)
        ('c', 'k')
        ('d', 'j')
        ('g', 'j')
        >>> # returns an exhaustive set if there are more combos requested than can be made uniquely
        >>> for sample in RandomCombination(10, 2, 3): print(sample)
        (0, 1)
        (0, 2)
        (1, 2)
        
    Note that the type of whatever is contained in the sequence like object must support ordering.
    """
    
    population: BasicSequenceProtocol
    
    int_population: Sequence[int]
        
    def __init__(self, population: Union[int, BasicSequenceProtocol], combo_length: int, number_of_combos: int):
        """
        :param population: The population to choose from.  If specified as an integer then the population will be
                           range(int).
        :param combo_length: The length for each combination as an integer
        :param number_of_combos: the number of unique combinations you want as an integer
        """

        if isinstance(population, int):
            n_population = population
            self.population = range(population)
            self.int_population = self.population
        else:
            n_population = len(population)
            self.int_population = range(n_population)
            self.population = population

        self.combo_length = combo_length

        self.possible_combos = int(comb(n_population, self.combo_length))

        self.number_of_combos = number_of_combos

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """
        Generate random combinations.

        If the number of requested combinations is greater than or equal to the number of possible combinations,
        this method will return all possible combinations. Otherwise, it will generate unique random combinations.

        :return: An iterator of tuples, where each tuple is a combination of elements from the population.
        """

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
