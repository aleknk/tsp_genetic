import numpy as np
from joblib import Parallel, delayed

from pytsp.population import Population
from pytsp.individual import Individual
from pytsp.selection import select_from_population

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from typing import Tuple, Optional

class TSPGA:
    """
    A class to solve the Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA).
    """

    def __init__(self, distance_matrix: np.ndarray) -> None:    
        self.distance_matrix = distance_matrix
        
    def run(self, 
            population_size: int, 
            num_generations: int,
            tournament_size: int = 5,
            crossover_method: str = "MP", 
            mutation_rate: float = 0.25, 
            num_swaps: int = 2,
            max_no_convergence: int = 10) -> Tuple[Individual, np.ndarray]:
        
        """
        Runs the genetic algorithm to solve the TSP problem.

        Parameters:
        Parameters:
        - population_size (int): The size of the population for each genetic algorithm run.
        - num_generations (int): The number of generations to evolve the population.
        - tournament_size (int, optional): The number of individuals competing in each tournament if tournament selection is used. Default is 5.
        - crossover_method (str, optional): The method used for crossover. Default is "MP".
        - mutation_rate (float, optional): The probability of a mutation occurring. Default is 0.25.
        - num_swaps (int, optional): The number of swaps in mutation if mutation is applied. Default is 2.
        - max_no_convergence (int, optional): The maximum number of generations without improvement before stopping. Default is 10.


        Output:
        - best_individual (Individual): The best individual found by the genetic algorithm.
        """

        # Instantiate Population
        tour_length = self.distance_matrix.shape[0]
        population = Population(population_size, tour_length)
        
        # Initialize population
        population.initialize_population()
        
        # Set distance matrix
        population.set_distance_matrix(self.distance_matrix)

        # Run the algorithm for the specified number of generations
        fitness_values = []

        # Variable that counts the number of consecutive generations without improvement
        count = 0
        
        # Starting with no best_individual
        self.best_individual = None
        
        # Main generation loop
        for _ in range(num_generations):
            
            # Calculate fitness of individuals in the population
            population.calculate_population_fitness()
            
            # Store best individual in the generation
            best_individual_generation = population.get_best_individual()
            best_individual_generation.calculate_fitness(self.distance_matrix)
            
            # Store best fitness of generation
            fitness_values.append(best_individual_generation.fitness)

            # Store best individual generation if best and check for no improvement
            if self.best_individual is not None:
                if best_individual_generation.fitness < self.best_individual.fitness:
                    self.best_individual = best_individual_generation
                    count = 0
                else:
                    count += 1
                if count == max_no_convergence:
                    break
            else:
                self.best_individual = best_individual_generation
            
            # Perform selection to choose parents for reproduction
            selected_individuals = select_from_population(population, tournament_size=tournament_size)

            # Create offspring through crossover
            descendence = population.crossover(selected_individuals, method=crossover_method)

            # Replace the current population with the descendence
            population.individuals = descendence
            
            # Mutate the new population
            population.mutate_population(mutation_rate, num_swaps)

        # 1D array representing fitness value of best individual in each generation
        self.fitness_values = np.array(fitness_values)
        
        return self.best_individual, self.fitness_values

    def run_multi(self, 
                  population_size: int, 
                  num_generations: int, 
                  tournament_size: int = 5, 
                  crossover_method: str = "MP", 
                  mutation_rate: float = 0.25, 
                  num_swaps: int = 2, 
                  max_no_convergence: int = 10,
                  n_starts: int = 10, 
                  n_jobs: int = 1) -> Tuple[Individual, np.ndarray]:
        
        """
        Executes multiple runs of the genetic algorithm in parallel and returns the best solution.

        Parameters:
        - population_size (int): The size of the population for each genetic algorithm run.
        - num_generations (int): The number of generations to evolve the population.
        - selection_method (str, optional): The method used to select parents. Default is "tournament".
        - tournament_size (int, optional): The number of individuals competing in each tournament if tournament selection is used. Default is 5.
        - crossover_method (str, optional): The method used for crossover. Default is "MP".
        - mutation_rate (float, optional): The probability of a mutation occurring. Default is 0.25.
        - num_swaps (int, optional): The number of swaps in mutation if mutation is applied. Default is 2.
        - max_no_convergence (int, optional): The maximum number of generations without improvement before stopping. Default is 10.
        - n_starts (int, optional): The number of times to run the algorithm from different random initializations. Default is 10.
        - n_jobs (int, optional): The number of CPU cores to use for parallel runs. Default is 1.

        Returns:
        tuple: The best individual (solution) found and an array of fitness values across all runs.
        """

        # Generate a list of inputs for each run
        inputs = [(population_size, num_generations, tournament_size, crossover_method, mutation_rate, 
                   num_swaps, max_no_convergence) for _ in range(n_starts)]
        
        # Execute the GA runs in parallel
        parallel_results = Parallel(n_jobs=n_jobs)(delayed(self.run)(*input) for input in inputs)
        
        # Extract the best individuals from each run
        best_individuals = [x[0] for x in parallel_results]
        
        # Select the best individual overall (i.e., the one with the lowest fitness value)
        self.best_individual = min(best_individuals, key=lambda x: x.fitness)
    
        # Compute the length of the longest fitness array
        max_length = max(len(fitness_values) for _, fitness_values in parallel_results)
        
        # Pad all fitness arrays to the length of the longest one, using NaN values for padding
        padded_arrays = [np.pad(fitness_values, (0, max_length - len(fitness_values)), constant_values=np.nan) 
                         for _, fitness_values in parallel_results]
        
        # Stack all padded arrays vertically to form a matrix. 
        self.fitness_values = np.vstack(padded_arrays)
            
        return self.best_individual, self.fitness_values

    def plot_fitness_curves(self, ax: Optional[Axes] = None) -> None:
        """
        Plots the fitness curves of the genetic algorithm runs.
        
        This method visualizes the performance of the genetic algorithm across generations. 
        If multiple runs are conducted (i.e., multiple fitness curves are available), 
        each will be plotted in a different color.
        """

        if ax is None:
            # Create the figure and axis for the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            show = True
        else:
            show = False
            
        # Check if we have multiple fitness curves (i.e., from multiple runs)
        if len(self.fitness_values.shape) == 2:
            # Extract the dimensions of the fitness_values matrix
            n, m = self.fitness_values.shape

            # Generate a unique color for each run using the seaborn color palette
            colors = sns.color_palette("husl", n)

            # Iterate over each run and plot its fitness curve
            for i in range(n):
                ax.plot(range(m), self.fitness_values[i], color=colors[i], label=f"Run {i + 1}")

            # Set the title for the plot with multiple runs
            ax.set_title('Fitness Curves for Multiple Runs')

        # If we have only one fitness curve
        else:
            # Plot the single fitness curve
            ax.plot(range(len(self.fitness_values)), self.fitness_values)
            
            # Set the title for the plot with a single run
            ax.set_title('Fitness Curve for Single Run')

        # Set the x and y axis labels
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Value')

        if show:
            # Display the plot
            plt.show()