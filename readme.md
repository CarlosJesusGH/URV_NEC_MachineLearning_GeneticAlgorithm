# General description
Optimization with Genetic Algorithms

### Objective
Implementation of a genetic algorithm (GA) for the clustering of the nodes of a graph, by means of the optimization of modularity.

### Genetic algorithm
Given the network G, any vector S may be seen as the chromosome corresponding to a partition in two clusters, and the fitness is related to its modularity Q(S). The objective is the implementation of a genetic algorithm to obtain the partition which maximizes modularity.
### Fitness
Modularity cannot be used directly as the fitness since it may take negative values, and also the difference in modularity of good partitions may be very small. You should define a transformation of modularity into an adequate positive fitness function, use the rank, or try with robust selection methods.

### Data
* Input: a network in Pajek format (*.net) (http://pajek.imfm.si/)
* Output: the highest modularity partition found, in Pajek format (*.clu), and its corresponding value of modularity

### Parameters
Try different variants of the genetic algorithm and of their corresponding parameters, for instance:
* Fitness function
* Selection schemes: proportional (roulette-wheel), truncation, ranking, tournament, fitness uniform (FUSS), etc.
* Crossover: one-point crossover, two-point crossover, uniform crossover, etc.
* Mutation
* Elitism
