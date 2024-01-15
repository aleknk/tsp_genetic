# PyTSPGA - A Python library for solving the Traveler Salesman Problem using Genetic Algorithms.

## **Instalation**

To utilize the PyTSP package, users must have a Python environment with Python 3.12 or later. The installation process can be initiated with the following steps:

1. Create a new Python environment:
   `conda create -n <your_env_name> python==3.12`

2. Activate the created environment:
   `conda activate <your_env_name>`

3. Install the required dependencies from the requirements.txt file:
   `pip install -r requirements.txt`

4. Finally, install the PyTSP package:
   `pip install .`

## **Usage**

### **Setting up the scenario**

The PyTSP package includes functionalities for generating and visualizing city distributions on the X-Y plane:

- `pytsp.cities.generate_cities`: Generates a random distribution of cities.
- `pytsp.cities.plot_cities`: Provides visualization of the city distribution.

For detailed information on these functions, refer to the `notebooks/create_cities.ipynb` notebook.

### **Running the Genetic Algorithm**

To find the optimal route, PyTSP offers two main functions within the `pytsp.tspga.TSPGA` class:

- `TSPGA.run`: Executes the Genetic Algorithm with specified parameters.
- `TSPGA.run_multi`: Similar to TSPGA.run, but utilizes a multi-start approach, running multiple instances of the Genetic Algorithm in parallel.

For an in-depth guide, see `notebooks/run_ga.ipynb` and `notebooks/multirun_ga.ipynb`.

### **Command Line Tools**

PyTSPGA includes two command-line tools for running the Genetic Algorithm, as defined in the setup.py file:

1. Single Run: `tspga_run --cities <path_to_cities_array> --save-dir <path_to_desired_output_dir>`

2. Multi-Run: `tspga_multirun --cities <path_to_cities_array> --n-starts 10 --save-dir <path_to_desired_output_dir>`

Both tools support various parameters. For detailed usage instructions, execute `tspga_run --help` or `tspga_multirun --help`, or refer to the aforementioned notebooks.
