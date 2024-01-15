import random
from pytsp.individual import Individual

def get_child(parent1,parent2,method="OX"):
    """
    Generates a child Individual using the specified crossover method.

    Parameters:
    - parent1 (Individual): The first parent Individual.
    - parent2 (Individual): The second parent Individual.
    - method (str): The crossover method to be used. (Default: "MP")
    - n (int): Number of crossover points to be used if method == "MP.  (Default: 3)

    Returns:
    - child (Individual): The child Individual obtained by crossover.
    """
    if method == "OX":
        return order_crossover(parent1,parent2)
    if method == "CX":
        return cycle_crossover(parent1,parent2)


def cycle_crossover(parent1, parent2):
    """
    Performs Cycle Crossover (CX) between two parent Individuals.

    Parameters:
    - parent1 (Individual): The first parent Individual.
    - parent2 (Individual): The second parent Individual.

    Returns:
    - child (Individual): The child Individual obtained by Cycle Crossover.
    """
    tour1, tour2 = parent1.tour, parent2.tour
    size = len(tour1)
    child_tour = [None] * size

    # Start from the first index
    start_index = 0
    
    while None in child_tour:
        # If this index is already filled, find the next empty spot
        if child_tour[start_index] is not None:
            start_index = child_tour.index(None)

        current_index = start_index

        # While the cycle hasn't looped back to the start index
        while child_tour[current_index] is None:
            child_tour[current_index] = tour1[current_index]
            current_index = tour2.index(tour1[current_index])

    return Individual(child_tour)

def order_crossover(parent1, parent2):
    """
    Performs Order Crossover (OX) between two parent Individuals.

    Parameters:
    - parent1 (Individual): The first parent Individual.
    - parent2 (Individual): The second parent Individual.

    Returns:
    - child (Individual): The child Individual obtained by Order Crossover.
    """
    tour1, tour2 = parent1.tour, parent2.tour
    size = len(tour1)
    child_tour = [None] * size

    # Randomly select two crossover points
    start, end = sorted([random.randint(0, size-1), random.randint(0, size-1)])
    
    # Copy the genes from parent1 between the two crossover points
    child_tour[start:end+1] = tour1[start:end+1]

    pos = (end + 1) % size
    for gene in tour2:
        if gene not in child_tour:
            child_tour[pos] = gene
            pos = (pos + 1) % size

    return Individual(child_tour)