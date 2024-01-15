import numpy as np
import random
from typing import List

class Individual:
    """
    Represents an individual in a genetic algorithm population.
    """

    def __init__(self, tour: List[int]):
        """
        Initializes a new individual with a given tour.

        Parameters:
        - tour (list): A list representing the order of cities in the tour.
        """
        self.tour = tour
        
    def calculate_fitness(self, distance_matrix: np.ndarray) -> None:
        """
        Calculates the fitness of the individual based on a given distance matrix.

        Parameters:
        - distance_matrix (numpy.ndarray): A 2D array representing the distances between cities.

        Output:
        - None (updates the fitness attribute of the individual object)
        """
        # Convert the tour into a numpy array
        tour_array = np.array(self.tour)

        # Shift the tour array by one position to the left
        tour_shifted = np.roll(tour_array, -1)

        # Calculate the sum of distances between consecutive cities in the tour using the distance matrix
        self.fitness = np.sum(distance_matrix[tour_array, tour_shifted])

        # Add the distance from the last city back to the first city (city 0)
        self.fitness += distance_matrix[tour_array[-1], tour_array[0]]
        

    def mutate(self, num_swaps: int = 1) -> None:
        """
        Mutates the individual by swapping a given number of cities in the tour,
        excluding the first city.

        Parameters:
        - num_swaps (int): The number of city swaps to perform (default is 1).

        Output:
        - None (updates the tour attribute of the individual object)
        """
        
        # Generate a list of unique random indices (excluding the first city) to swap
        indices = random.sample(range(1, len(self.tour)), num_swaps)

        # Perform the swaps
        for i in range(num_swaps):
            index1 = indices[i]
            index2 = random.choice(list(set(range(1, len(self.tour))) - set([index1])))

            # Swap the cities at index1 and index2 in the tour
            self.tour[index1], self.tour[index2] = self.tour[index2], self.tour[index1]
