import numpy as np
import warnings
import time
warnings.filterwarnings('error')
np.random.seed(1)
# -----------------
# Mark start training time
startTime = time.clock()
# ------------------------------------------------------------------------------
def createAdjacenceMatrix(fileName):
    isEdges = False
    lineCounter = 0
    edges = []
    with open(fileName) as infile:
        for line in infile:
            if isEdges:
                # Remove blank spaces
                line = " ".join(line.split())
                edge = line.split(' ')
                if len(edge) > 2:
                    edge = [edge[0], edge[1]]
                    edge = [int(i) for i in edge]
                    edges.append(edge)
            else:
                lineCounter += 1
            #
            if not isEdges and line.startswith("*Edges"):
                isEdges = True
                matrixSize = lineCounter - 2
    # -----------------------------------------------------
    adjMat = np.zeros((matrixSize,matrixSize))
    for edge in edges:
        # Replicate the connection in both ways
        # At the end adjacency matrix should be simmetric
        adjMat[edge[0] - 1,edge[1] - 1] = 1
        adjMat[edge[1] - 1,edge[0] - 1] = 1
    return adjMat, matrixSize
# ------------------------------------------------------------------------------
def createInitialPopulation(numIndividuals, codeSize):
    population =  np.random.randint(2, size=(numIndividuals, codeSize))
    # Make a threshold over 0.5
    population[population > 0.5] = 1
    population[population < 0.5] = 0
    return np.matrix(population)
# ------------------------------------------------------------------------------
def computeModularity(matrix, pop, L, Ki, Kj, codeSize):
    modularities = np.zeros(len(pop))
    # iterate individuals and then positions of adjacency matrix
    for ind,individual in enumerate(pop):
        individual = individual.getA1() # Return self as a flattened ndarray.
        modularity = 0
        for j in range(codeSize):
            for i in range(codeSize):
                Si = individual[i]
                Sj = individual[j]
                sameCluster = Si*Sj+(1-Si)*(1-Sj)   # gamma
                modularity +=  (matrix[i][j]-(Ki[i]*Kj[j]/L)) * sameCluster    # We don't need the number 2 because we are using a symmetric matrix
                # print "sameCluster in [%d][%d] = %f... modularity= %f" %(i, j, sameCluster, modularity)
        modularity = modularity * (1/L) # We don't need the number 2 because we are using a symmetric matrix
        modularities[ind] = modularity
    return modularities
# ------------------------------------------------------------------------------
def computeFitnessOld(modularities, popSize):
    # print "computeFitness"
    # Make all values positive and between 0 and 1
    fitness = (modularities + 1)/2
    # print fitness
    # Equalize fitness
    min = np.amin(fitness);
    max = np.amax(fitness);
    limMax = 1;
    limMin = 0.01;
    # Force values to go from 0.01 to 1
    try:
        fitnessAux = (((limMax-limMin) * (fitness - min)) / (max - min)) + limMin;
        # Giving more separability to best results
        # This line make results better but execution time is increased a lot
        # fitnessAux = (fitnessAux * 10)**2
        # Force values to sum 1 (this will be used in numpy.random.choice function)
        fitness = fitnessAux/np.sum(fitnessAux)
    except Warning as w:
        # print "Warning: all values are the same"
        fitness = np.full(popSize, float(1)/popSize)
        # print fitnessAux
    return fitness
# ------------------------------------------------------------------------------
def computeFitness(modularities, popSize):
    # Get index order, if all values are the same it returns [1,2,3,4...]
    fitness = np.argsort(np.argsort(modularities))
    # Force values to sum 1 (this will be used in numpy.random.choice function)
    fitness = fitness/float(np.sum(fitness))
    return fitness
# ------------------------------------------------------------------------------
def createNextGeneration(population, fitness, popSize, codeSize, mutationProb, elitismSize):
    children = []
    # Include elite individuals
    if (elitismSize > 0):
        elitismIndices = fitness.argsort()[-1*elitismSize:]
        for i in range(elitismSize):
            children.append(population[elitismIndices[i]].getA1())
    # Breed to complete the population
    while len(children) < popSize:
        fatherInd = getParentAccordingToFitness(population, fitness, popSize, codeSize)
        motherInd = getParentAccordingToFitness(population, fitness, popSize, codeSize)
        if fatherInd != motherInd:
            father = population[fatherInd].getA1()
            mother = population[motherInd].getA1()
            child = [0] * codeSize
            if True:   # Uniform crossover
                # Mix chromosomes using uniform crossover
                crossoverSelector = np.random.randint(2, size=codeSize)
                for i in range(codeSize):
                    child[i] = father[i] if (crossoverSelector[i] == 1) else mother[i]
            if False:    # One point crossover
                fatherCodeSize = codeSize / 2
                motherCodeSize = codeSize / 2
                # Add 1 chromosome to father if odd number
                if (motherCodeSize + fatherCodeSize) != codeSize:
                    fatherCodeSize += 1
                child = np.hstack((father[:fatherCodeSize],mother[codeSize-motherCodeSize:]))
            # ------------------------
            # Add mutation, Select random position in code and change it
            mutate = np.random.rand(1)
            if mutate < mutationProb:
                randPos = np.random.randint(codeSize)
                child[randPos] = 1 if child[randPos] == 0 else 0
            # ------------------------
            children.append(child)
    return np.matrix(children)
# ------------------------------------------------------------------------------
def getParentAccordingToFitness(population, fitness, popSize, codeSize):
    # print "getParentAccordingToFitness"
    choice = np.random.choice(popSize, 1, p=fitness)
    # print choice
    return choice
# ------------------------------------------------------------------------------
def printCluFile(individual, codeSize, fileName, popSize, elitismSize, numGenerations, mutationProb, bestModularity):
    outName = fileName.split('/')[-1]
    outName = outName.split('.')[0]
    outName = '../outputs/' + outName + '_out'
    outName += '_[popS=%d, elitS=%d, numGen=%d, mutProb=%.2f, bestMod=%.3f]' % (popSize, elitismSize, numGenerations, mutationProb, bestModularity)
    outName += '.clu'
    f = open(outName, 'w')
    f.write('*Vertices ' + str(codeSize) + '\n')
    for i in range(codeSize):
        f.write(str(individual[i]) + '\n')
    f.close()
