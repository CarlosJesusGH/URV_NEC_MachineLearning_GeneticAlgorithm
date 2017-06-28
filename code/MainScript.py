# cd '/home/cj/Dropbox/Personal/Study/MasterDegreeArtificialIntelligence/1st Semester/NEC/Exercise 4/implementation' && source ~/Coding/venv/bin/activate && python main_script.py
from GeneticAlgorithm import *
import os

# Settings
# inputFile = "../networks/graph3+1+3.net"
# inputFile = "../networks/graph3+2+3.net"
# inputFile = "../networks/20x2+5x2.net"
# inputFile = "../networks/circle9.net"
# inputFile = "../networks/rb25.net"
# inputFile = "../networks/256_4_4_2_15_18_p.net"
# inputFile = "../networks/cliques_line.net"
# inputFile = "../networks/grid-6x6.net"
# inputFile = "../networks/rhesus_simetrica.net"
# inputFile = "../networks/256_4_4_4_13_18_p.net"   # Size 10/10
# inputFile = "../networks/clique_stars.net"
# inputFile = "../networks/grid-p-6x6.net"
inputFile = "../networks/star.net"
# inputFile = "../networks/adjnoun.net"
# inputFile = "../networks/dolphins.net"
# inputFile = "../networks/qns04_d.net"
# inputFile = "../networks/wheel.net"
# inputFile = "../networks/cat_cortex_sim.net"
# inputFile = "../networks/rb125.net"
# inputFile = "../networks/zachary_unwh.net"
#
popSize = 100 #100 or 50
elitismSize = int(popSize * 0.2)
numGenerations = 200
mutationProb = 0.25
modularityHistory = []
verbose = 1
restingTime = 0.25

# ------------------------------------------------------------------------------
# Start Script
# ------------------------------------------------------------------------------
# Create adjacency matrix from input file
adjMatrix, codeSize = createAdjacenceMatrix(inputFile)
# Compute L, Ki and Kj
L = np.sum(adjMatrix)
Ki = np.sum(adjMatrix, axis=0)
Kj = np.sum(adjMatrix, axis=1)
if verbose > 1: print "Adjacency matrix"; print adjMatrix
# ---------------
# Create initial population using random numbers
population = createInitialPopulation(numIndividuals=popSize, codeSize=codeSize)
if verbose > 1: print "Population"; print population
# ---------------
# Compute modularity
modularities = computeModularity(matrix=adjMatrix, pop=population, L=L, Ki=Ki, Kj=Kj, codeSize=codeSize)
if verbose > 0: print "Modularities"; print modularities
modularityHistory.append(modularities)
# ---------------
# Get fitness from modularity
fitness = computeFitness(modularities=modularities, popSize=popSize)
if verbose > 0: print "Fitness"; print fitness
# ---------------
# Iterate for future generations
bestModularity = np.max(modularities)
lastLoopBest = bestModularity
bestIndividual = population[np.argmax(modularities)]
for i in range(numGenerations):
    if i%10 == 0 or i == (numGenerations-1):
        print "Generation= %d" % i
        print '[popS=%d, elitS=%d, numGen=%d, mutProb=%f, bestMod=%.3f]' % (popSize, elitismSize, numGenerations, mutationProb, lastLoopBest)
    # Get next generation
    population = createNextGeneration(population=population, fitness=fitness, popSize=popSize, codeSize=codeSize, mutationProb=mutationProb, elitismSize=elitismSize)
    if verbose > 0: print "newGeneration"; print population
    # ---------------
    # Compute modularity
    modularities = computeModularity(matrix=adjMatrix, pop=population, L=L, Ki=Ki, Kj=Kj, codeSize=codeSize)
    if verbose > 0: print "Modularities"; print modularities
    modularityHistory.append(modularities)
    # ---------------
    # Get fitness from modularity
    fitness = computeFitness(modularities=modularities, popSize=popSize)
    if verbose > 0: print "Fitness"; print fitness
    # Save best individual
    lastLoopBest = np.max(modularities)
    if lastLoopBest > bestModularity:
        bestModularity = lastLoopBest
        bestIndividual = population[np.argmax(modularities)]
    # Make a pause before starting new cicle for cooling processor
    time.sleep(restingTime)
# ---------------
# Create CLU file for Pajek
printCluFile(bestIndividual.getA1(), codeSize, inputFile, popSize, elitismSize, numGenerations, mutationProb, bestModularity)
# ---------------
# Print final history
if verbose > 0: print "==============\nFinal modularities and sum by row\n==================="
if verbose > 0: print np.matrix(modularityHistory)
if verbose > 0: print np.sum(modularityHistory, axis=1).T
# print "==============\nBest value\n==================="
# print np.max(modularityHistory)
print "==============\nBest modularity\n==================="
print bestModularity
print "==============\nBest individual\n==================="
print bestIndividual
print "==============\nMeasured time\n==================="
measuredTime = time.clock() - startTime
print '%.2f - (%.2f * %d) = %.2f' % (measuredTime, restingTime, numGenerations, measuredTime-(restingTime*numGenerations))
# -----------------
os.system("spd-say 'CJ, I finished your simulation'")
