<!-- omit in toc -->
# Symbolic Regression using Genetic Programming
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Contributors](https://img.shields.io/badge/Contributors-4-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)

## Disclaimer 
 **I worked on this project together with my colleagues, which are listed in the Contributors section. This is the official where we worked together:[CI2024_Project](https://github.com/FerraiuoloP/CI2024_Project).**

### Program flow
- **Initialization:** We start by creating the population. Contrary to the literature where a ramped half-and-half initialization is prefered, we decided to a grow/full ratio of 0.95/0.05. This is because we found that the grow method was more effective than the full method as the latter tends to create very large trees that lead to a bloating phenomenon. 
An island model is used to promote diversity so this initialization is repeated for each island.
  
- **Evolution loop:** For each island, the algorithm proceeds independently. If a migration occurs, a random island is selected where a random individual migrates to a random island. A new population is then  generated at each iteration based on a 0.95/0.05 crossover/mutation ratio. Fitness is computed for every offsprings,they are added to the population and the is sorted. Then the least fit individuals are removed. This process is repeated until a certain number of generations has been reached. If a takeover is detected at the end of an iteration, from the next iteration a new population replaces the old one. This is done to avoid being stuck in a local optima.
- 
## Description
This repository contains an implementation of a Symbolic Regressor using a Tree-based Genetic Algorithm. The goal is to regress a formula from a given dataset, by evolving a population of mathematical formulas represented as trees.At each iteration, the population is either mutated or crossed over to generate new individuals. The fitness of each individual is evaluated by computing the Mean Squared Error (MSE) between the predicted values and the actual values of the dataset.

## Choices rationale
- **Constraint satisfaction:** Every tree generated is guaranteed to be valid. That is, every variable appears in said three and its fitness isn't NaN/Inf(on the training set). 
Although this negatively impacts the performance of the algorithm and efficiency of mutation/crossover, this ensures that the algorithm will always converge to a valid solution.

- **Fitness function:** We decided to use the mean squared error as our fitness function. We tried other functions such as the mean absolute error and the root mean squared error, but the mean squared error was the best. (The code for other fitness functions has been removed)
- **Parent selection:** We found that linear ranking worked best for our scenario. I tried many other approaches such as fitness proportional selection, tournament selection and exponential ranking, but linear ranking was the best.
- **Survival selection:** Deterministic. Only the fittest individuals survive.
- **Crossover:** We use a classical crossover where we exchange two different subtrees.
- **Mutation:** We use a subtree mutation or single node mutation with 0.5 probability. Other mutations such as the hoist mutation and the point mutation weren't tried as I deemed them useless for our scenario.
- **Crossover-Mutation ratio:** We found that a 65-35 ratio worked best for our scenario. I tried many ratio but none were as good as this one. For mutation, the 0.5 probability for subtree/single node worked best.
- **Takeover:** I use a classical takeover detection mechanism where a takeover is triggered once 90% of the population has the same fitness. Once triggered, this population gets replaced by a new one in the same amount. 0.9 has been found to be the best threshold for our scenario.
- **Islands:** We decided to promote diversity by using an island architecture. Population is divided into islands. Randomly, an individual may migrate from an island to another. In this way, we are exploring different areas of the search space at the same time.
Through trial and error a low migration rate of 0.001 was found to be the best. If we set it too high, the population will be contaminated by the same individuals and the algorithm will converge to a local optima too son.
- **Elitism:** We decided not to employ any elitism. We found that it didn't bring any benefit to our scenario as it would converge to a local optima too soon, just like a high migration rate would do
## Project Structure
The project is organized as follows:
- `src/sym_reg.ipynb`
  - Jupyter notebook file containing the driver code of the symbolic regressor. Once the algorithm stops, it will output the best formula found by the algorithm
  ```python
  np.sin(x[0]) # a simple mathematical formula for problem 1
  ```
- `src/tree.py`
  - File containing the class definition of the tree structure used to represent the mathematical formulas. The class contains methods to evaluate the tree, mutate it, and cross it over with another tree.
- `data/`
  - A folder that contains eight different input problems (*npz* files). Each problem is represented by a dataset that can be used to train and test the model.
- `pyproject.toml`
  - The configuration file containing dependencies and metadata for the project. *Poetry* is used for package management.

## Other Contributers
<table>
  <tr>
    <td align="center" style="border: none;">
      <a href="https://github.com/AgneseRe">
        <img src="https://github.com/AgneseRe.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>AgneseRe</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/FerraiuoloP">
        <img src="https://github.com/FerraiuoloP.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>FerraP</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GDennis01">
        <img src="https://github.com/GDennis01.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>GDennis01</sub>
      </a>
    </td>
  </tr>
</table>

## License
This project is licensed under the MIT License.