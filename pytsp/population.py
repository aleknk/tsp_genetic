import numpy as np
import random

from pytsp.individual import Individual
from pytsp.crossover import get_child

from typing import List, Optional

class Population:
    """
    Represents a population of individuals in a genetic algorithm.
    """

    def __init__(self, population_size: int, tour_length: int) -> None:
        self.population_size: int = population_size
        self.tour_length: int = tour_length
        self.individuals: List[Individual] = []
        self.individuals_fitnesses: List[float] = []
        self.distance_matrix: Optional[np.ndarray] = None
        
    def initialize_population(self, seed: int = 42) -> None:
        """
        Initializes the population by generating random individuals.

        Output:
        - None (updates the individuals attribute of the population object)
        """
        np.random.seed(seed)
        for _ in range(self.population_size):
            # Generate a random tour by permuting the cities
            tour = np.random.permutation(self.tour_length-1)+1
            tour = np.insert(tour,0,0)
            
            # Create a new individual with the tour and add it to the population
            individual = Individual(tour.tolist())
            self.individuals.append(individual)
        np.random.seed()
        
    def set_distance_matrix(self, distance_matrix: np.ndarray) -> None:
        """
        Sets the distance matrix for calculating fitness.

        Parameters:
        - distance_matrix (numpy.ndarray): A 2D array representing the distances between cities.

        Output:
        - None (updates the distance_matrix attribute of the population object)
        """
        self.distance_matrix = distance_matrix

    def calculate_population_fitness(self) -> None:
        """
        Calculates the fitness of each individual in the population.

        Output:
        - None (updates the individuals_fitnesses attribute of the population object)
        """
        for individual in self.individuals:
            individual.calculate_fitness(self.distance_matrix)

        self.individuals_fitnesses = [individual.fitness for individual in self.individuals]

    def crossover(self, selected_individuals: List[Individual], method: str = "OX") -> List[Individual]:
        """
        Performs crossover between selected individuals to create offspring for the next generation.

        Parameters:
        - selected_individuals (list): A list of selected individuals for crossover.
        - n (int): Number of crossover points to perform if method=="MP". (Default to 1)

        Output:
        - offspring (list): A list of offspring individuals.
        """
        offspring = []
        for _ in range(self.population_size):
            # Randomly select two parents
            parent1, parent2 = random.sample(selected_individuals, 2)

            # Perform crossover to create a child
            child = get_child(parent1,parent2,method=method)

            # Add the child to the offspring
            offspring.append(child)

        return offspring

    def mutate_population(self, mutation_rate: float, num_swaps: int = 2) -> None:
        """
        Mutates the individuals in the population based on a given mutation rate.

        Parameters:
        - mutation_rate (float): The probability of mutation for each individual.
        - num_swaps (int): The number of city swaps to perform during mutation (default is 2).

        Output:
        - None (updates the tour attribute of the individual objects in the population)
        """
        for individual in self.individuals:
            if np.random.rand() < mutation_rate:
                individual.mutate(num_swaps=num_swaps)

    def get_best_individual(self) -> Individual:
        """
        Returns the best individual in the population.

        Output:
        - best_individual (Individual): The best individual.
        """
        return min(self.individuals, key=lambda x: x.fitness)

    def get_best_fitness(self) -> float:
        """
        Returns the fitness of the best individual in the population.

        Output:
        - best_fitness (float): The fitness of the best individual.
        """
        return min([individual.fitness for individual in self.individuals])

