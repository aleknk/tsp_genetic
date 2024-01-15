import numpy as np
from typing import List
from pytsp.individual import Individual
from pytsp.population import Population 

def select_from_population(population: Population, method: str = "tournament", tournament_size: int = 10) -> List[Individual]:
    """
    Selects individuals from a population using the specified selection method.

    Parameters:
    - population (Population): The population from which Individuals are selected.
    - method (str): The selection method to use. Options are 'roulette' or 'tournament'. (Default: 'tournament')
    - tournament_size (int): If the tournament method is selected, this determines the number of individuals
                             participating in each tournament. (Default: 10)

    Returns:
    - list[Individual]: The list of selected Individuals.

    Raises:
    - ValueError: If an unrecognized selection method is provided.
    """
    
    if method == 'tournament':
        return tournament_selection(population, tournament_size)
    else:
        raise ValueError(f"Unrecognized selection method: {method}")


def tournament_selection(population: Population, tournament_size: int) -> List[Individual]:
    """
    Performs Tournament Selection on a population of Individuals.

    Parameters:
    - population (Population): The population from which Individuals are selected.
    - tournament_size (int): The number of individuals participating in each tournament.

    Returns:
    - list[Individual]: The list of selected Individuals.
    """
    selected_individuals = []
    for _ in range(population.population_size):
        # Randomly select individuals for the tournament
        tournament_indices = np.random.choice(range(population.population_size), size=tournament_size, replace=False)
        tournament_individuals = [population.individuals[index] for index in tournament_indices]

        # Select the best individual from the tournament as the winner
        winner = min(tournament_individuals, key=lambda x: x.fitness)
        selected_individuals.append(winner)

    return selected_individuals
