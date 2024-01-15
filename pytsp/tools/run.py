import argparse
import numpy as np
import os

import matplotlib.pyplot as plt

from pytsp.tspga import TSPGA
from pytsp.cities import get_distance_matrix, plot_cities

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--cities", type=str, required=True, help="Path to numpy matrix indicating the cities positions on x-y plane.")
    parser.add_argument("--population-size", type=int, default=100, help="Size of the population.")
    parser.add_argument("--num-generations", type=int, default=1000, help="Number of generations to run.")
    parser.add_argument("--tournament-size", type=int, default=10, help="Tournament size.")
    parser.add_argument("--crossover", type=str, default="OX", choices=["OX", "CX"], help="Crossover method to use.")
    parser.add_argument("--mutation-rate", type=float, default=0.01, help="Mutation rate.")
    parser.add_argument("--num-swaps", type=int, default=1, help="Number of swaps for mutation.")
    parser.add_argument("--max-no-convergence", type=int, default=200, help="Maximum number of generations with no convergence.")
    parser.add_argument("--save-dir", type=str, default="./", help="Directory to save the resulting plot.")

    args = parser.parse_args()

    cities = np.load(args.cities)
    distance_matrix = get_distance_matrix(cities)
    
    genetic = TSPGA(distance_matrix)
    
    best_individual, fitnesses = genetic.run(population_size=args.population_size, 
                                             num_generations=args.num_generations, 
                                             tournament_size=args.tournament_size, 
                                             crossover_method=args.crossover, 
                                             mutation_rate=args.mutation_rate, 
                                             num_swaps=args.num_swaps, 
                                             max_no_convergence=args.max_no_convergence)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
    genetic.plot_fitness_curves(ax=ax[0])
    plot_cities(cities=cities, tour=best_individual.tour, ax=ax[1], fitness=best_individual.fitness)
    
    plt.savefig(os.path.join(args.save_dir, "tspga_run.png"))

if __name__ == "__main__":
    main()