import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def generate_cities(n, min_val=0, max_val=100):
    """
    Generate a list of random city coordinates.

    Parameters:
    - n (int): The number of cities to generate.
    - min_val (int, optional): The minimum value for x and y coordinates. Default is 0.
    - max_val (int, optional): The maximum value for x and y coordinates. Default is 100.

    Returns:
    - numpy.ndarray: An array of shape (n, 2) containing the (x, y) coordinates of cities.

    Example:
    >>> generate_cities(5)
    array([[23, 67], 
           [78, 45], 
           [10, 89], 
           [34, 56], 
           [90, 12]])
    """
    
    # Generate an array of shape (n, 2) with random integers 
    # between min_val and max_val for each city's (x, y) coordinates
    points = np.random.randint(min_val, max_val, size=(n, 2))
    
    return points

def get_distance_matrix(cities):
    return distance_matrix(cities,cities)

def plot_cities(cities, start_city_idx=0, tour=None, ax=None, fitness=None):
    """
    Plot the given cities on a 2D plane. Optionally, a tour can be provided 
    to visualize the path connecting these cities.

    Parameters:
    - cities (numpy.ndarray or list): An array of shape (n, 2) containing (x, y) coordinates of cities.
    - start_city_idx (int, optional): Index of the starting city in the tour. This city is highlighted in red. Default is 0.
    - tour (list, optional): A list of city indices representing the order to visit. Default is None.
    - ax (matplotlib.axes.Axes, optional): An axes object to plot on. If None, a new figure and axes are created. Default is None.

    Example:
    >>> cities = np.array([[23, 67], [78, 45], [10, 89], [34, 56], [90, 12]])
    >>> plot_cities(cities, tour=[0, 1, 2, 3, 4])
    """

    # Convert cities to a DataFrame for easier plotting with seaborn
    cities_df = pd.DataFrame(cities, columns=["x", "y"])

    # Create a new plot if no axes is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
        show = True
    else:
        show=False

    # Plot all cities as black points
    if fitness is None:
        ax.set_title("Cities distribution")
    else:
        ax.set_title(f"Cities distribution || Fitness: {round(fitness,2)}")
        
    sns.scatterplot(data=cities_df, x="x", y="y", ax=ax, color='black')
    
    # Highlight the starting city in red
    sns.scatterplot(data=cities_df.iloc[start_city_idx:start_city_idx + 1], x="x", y="y", ax=ax, color='red', s=100)

    # If a tour is provided, plot the path connecting the cities in the tour
    if tour is not None:
        tour_array = np.array(tour)

        # Plot edges between consecutive cities in the tour
        for i in range(len(tour_array) - 1):
            city_idx1 = tour_array[i]
            city_idx2 = tour_array[i + 1]
            x1, y1 = cities[city_idx1]
            x2, y2 = cities[city_idx2]
            ax.plot([x1, x2], [y1, y2], color='lightgray')

        # Connect the last city in the tour back to the starting city
        city_idx1 = tour_array[-1]
        city_idx2 = tour_array[0]
        x1, y1 = cities[city_idx1]
        x2, y2 = cities[city_idx2]
        ax.plot([x1, x2], [y1, y2], color='lightgray')

    # Display the plot
    if show:
        plt.show()

    