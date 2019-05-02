#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import datetime
import copy

class Individual:
    def __init__(self, x=None, y=None):
        if x!=None and y!=None:
            self.x = x
            self.y = y
        else:
            self.x = To_Binary(random.randint(0, 1023))
            self.y = To_Binary(random.randint(0, 1023))

    def get_x(self):
        return copy.deepcopy(self.x)

    def get_y(self):
        return copy.deepcopy(self.y)

    def update_x(self, x):
        self.x = copy.deepcopy(x)

    def update_y(self, y):
        self.y = copy.deepcopy(y)

    def Fitness_x_y(self):
        x_i = To_Integer(self.x)
        y_i = To_Integer(self.y)
        fitness = -np.absolute(0.5 * x_i * np.sin(np.sqrt(np.absolute(x_i)))) -np.absolute(y_i * np.sin(30 * np.sqrt(np.absolute(x_i/y_i))))
        return fitness

    def Global_Optimum_Found(self):
        if To_Integer(self.get_x()) == 903 and To_Integer(self.get_y()) == 917: #if self.Fitness_x_y()<=(-1356.482683119401):
            return 1
        else:
            return 0

    def CEP(self, percentage=0.01):
        x = (self.Fitness_x_y()) - (-1356.482683119401) 
        y = x / -1356.482683119401
        CE = abs(y)        
        if CE <= percentage:
            return 1
        else:
            return 0

class Population:
    def __init__(self, population_size=None):
        self.population_list = []
        if population_size != None:
            for i in range(population_size):
                individual = Individual()
                self.population_list.append(individual)
        return
    
    def Add(self, individual):
        individual_copy = copy.deepcopy(individual)
        self.population_list.append(individual_copy)
        
    def Population_Length(self):
        return(len(self.population_list))

    def Average_Fitness(self):
        average_before_division = 0
        mean_before_div = []
        length = self.Population_Length()
        for i in range(length):
            average_before_division += self.population_list[i].Fitness_x_y()
            mean_before_div.append(self.population_list[i].Fitness_x_y())
        average = average_before_division/length
        summa = sum(mean_before_div)
        mean = summa / length
        return(mean)

    def Standard_Deviation(self):
        average_before_division = 0
        mean_before_div = []
        length = self.Population_Length()
        for i in range(length):
            average_before_division += self.population_list[i].Fitness_x_y()
            mean_before_div.append(self.population_list[i].Fitness_x_y())
        average = average_before_division/length
        summa = sum(mean_before_div)
        mean = summa / length
        temp = 0
        for value in mean_before_div:
            temp += (value - mean)**2
        std = math.sqrt(temp/len(mean_before_div))
        return std
    
    def Evaluate_Population(self):
        CEP_025 = 0
        CEP_01 = 0
        perfect = 0
        length = self.Population_Length()
        for i in range(length):
            CEP_025 += self.population_list[i].CEP(0.025)
            CEP_01 += self.population_list[i].CEP(0.01)
            perfect += self.population_list[i].Global_Optimum_Found()
        return(CEP_025, CEP_01, perfect)

def Mutation(population, Pm, bool_mutation):
    def Swap_Bit(bit):
        if bit == "0":
            return "1"
        else:
            return "0"
    if bool_mutation:
        for i in range(population.Population_Length()-1):
            individual_new = copy.deepcopy(population.population_list[i])
            x_coordinate = list(individual_new.get_x())
            y_coordinate = list(individual_new.get_y())
            leng = len(x_coordinate)
            for ii in range(leng):
                mutate_x = random.random()
                mutate_y = random.random()
                if mutate_x < Pm:
                    x_coordinate[ii] = Swap_Bit(x_coordinate[ii])
                if mutate_y < Pm:
                    y_coordinate[ii] = Swap_Bit(y_coordinate[ii])
            str_x = ''.join(x_coordinate)
            str_y = ''.join(y_coordinate)
            individual_new.update_x(str_x)
            individual_new.update_y(str_y)
            population.population_list[i] = individual_new
    return population

def Crossover(population, Pc, bool_crossover):
    if bool_crossover:
        i = 0
        while i < population.Population_Length()-2:
            make_crossover = random.random()
            if make_crossover < Pc:
                x_temp = copy.deepcopy(population.population_list[i].get_x())
                individual_1 = copy.deepcopy(population.population_list[i])
                individual_2 = copy.deepcopy(population.population_list[i+1])
                individual_1.update_x(individual_2.get_x())
                individual_2.update_x(x_temp)
                population.population_list[i] = individual_1
                population.population_list[i+1] = individual_1
            i += 2
    return population

def Selection(population, k=5):
    next_generation = Population()
    length=len(population.population_list)
    while len(next_generation.population_list) < length:
        best_ind = None
        for i in range(k):
            x = random.randint(0, length-1)
            ind = copy.deepcopy(population.population_list[x])
            if best_ind == None:
                best_ind = ind
            else:
                if best_ind.Fitness_x_y() > ind.Fitness_x_y():
                    best_ind = ind
        next_generation.Add(best_ind)
    return next_generation

def Genetic_Algorithm(population_size, generations, k_Tournament, Pc, bool_crossover, Pm, bool_mutation, val_av_korning):
    generation = 1
    evolution_of_average_fitness = []
    evolution_of_population_average_fitness = []
    found_optimum_first_on_generation_nr = 0 #set to a number when first found
    found_1_pecent_first_on_generation_nr = 0 #set to a number when first found
    found_25_percent_first_on_generation_nr = 0 #set to a number when first found
    population = Population(population_size)
    (found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr) = Found_On(population, generation, found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr)
    evolution_of_average_fitness.append(population.Average_Fitness())
    evolution_of_population_average_fitness.append(population.Average_Fitness())
    evolution_of_individuals_x = []
    evolution_of_individuals_y = []
    evolution_of_individuals_fitness = []
    average_fitness = []
    std_fitness = []
    successes_sum = 0

    EVALUATE_METRICS = Population()
    for ind in population.population_list:
        EVALUATE_METRICS.Add(ind)
    print("GENERATION: " + str(generation))
    while generation < generations:
        generation += 1
        plotting_fitness = []
        plotting_x = []
        plotting_y = []
        for individual in population.population_list:
            plotting_fitness.append(individual.Fitness_x_y())
            plotting_x.append(To_Integer(individual.get_x()))
            plotting_y.append(To_Integer(individual.get_y()))
            successes_sum += individual.Global_Optimum_Found()
        evolution_of_individuals_fitness.append(plotting_fitness)
        evolution_of_individuals_x.append(plotting_x)
        evolution_of_individuals_y.append(plotting_y)

        if val_av_korning != True:
            if generation == 10 or generation == 100 or generation == 1000:
                save_as= "_generation_" + str(generation)
                plot_evolution(np.asarray(evolution_of_individuals_x), np.asarray(evolution_of_individuals_y),save_as)

        #SELECTION
        population = copy.deepcopy(Selection(population, k_Tournament))
        
        #CROSSOVER
        population = copy.deepcopy(Crossover(population, Pc, bool_crossover))
        
        #MUTATION
        population = copy.deepcopy(Mutation(population, Pm, bool_mutation))
        
        for ind in population.population_list:
            EVALUATE_METRICS.Add(ind)
            length_evaluate = len(EVALUATE_METRICS.population_list)

        (found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr) = Found_On(population, generation, found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr)
        if generation%100==0:
            print("GENERATION: " + str(generation))
        
        if val_av_korning == True:
            evolution_of_average_fitness.append(EVALUATE_METRICS.Average_Fitness())
            evolution_of_population_average_fitness.append(population.Average_Fitness())
    #END OF WHILE-LOOP

    if val_av_korning == True:
        
        save_as= "_" + datetime.datetime.now().strftime("%m_%d_%H_%M")
        plt.plot(evolution_of_population_average_fitness)
        plt.savefig("evolution_of_population" + save_as)

        plt.plot(evolution_of_average_fitness)
        plt.savefig("evolution_of_average" + save_as)

        save_as= "_" + datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        plot_evolution(np.asarray(evolution_of_individuals_x), np.asarray(evolution_of_individuals_y),save_as)
    return (found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr, successes_sum)
#END GENETIC ALGORITHM


def Found_On(population, generation
    ,found_optimum_first_on_generation_nr
    ,found_1_pecent_first_on_generation_nr
    ,found_25_percent_first_on_generation_nr):

    CEP_025, CEP_01, perfect = population.Evaluate_Population()
    if perfect != 0 and found_optimum_first_on_generation_nr == 0: 
        found_optimum_first_on_generation_nr = generation
    if CEP_01 != 0 and found_1_pecent_first_on_generation_nr == 0:
        found_1_pecent_first_on_generation_nr = generation
    if CEP_025 != 0 and found_25_percent_first_on_generation_nr == 0:
        found_25_percent_first_on_generation_nr = generation
    return(found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr)

def Fitness_x_y(x, y):
    fitness = -np.absolute(0.5 * x * np.sin(np.sqrt(np.absolute(x)))) \
    -np.absolute(y * np.sin(30*np.sqrt(np.absolute(x/y))))
    return fitness

def To_Binary(x):
    x_temp = bin(x)    
    x_t = x_temp[2:]
    while len(x_t)<10:
        x_t = "0" + x_t
    return(x_t)
    
def To_Integer(x): #Correct with mapping
    y = (990*(int(x,2)/1023) + 10)
    z = math.ceil(y)
    return z

def plot_evolution(X, Y, name): #prickar i 3D
    Z = Fitness_x_y(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xmin, ymin = np.unravel_index(np.argmin(Z), Z.shape)
    mi = (X[xmin,ymin], Y[xmin,ymin], Z.min())
    ax.set_title("The Evolution of fitness")
    print("Obtained minimum value[X,Y,Z]: " + str(mi))
    print("The integer minimum value[X,Y,Z]: (903, 917, " + str(Fitness_x_y(903,917)) + ")")
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1500.0/float(DPI),1500.0/float(DPI))
    ax.scatter(X,Y,Z, zdir='z',s=10, cmap=cm.PRGn) #ax.scatter(X, Y, Z, rstride=1, cstride=1, cmap=cm.PRGn)    #plt.scatter(X, Y, Z, c=t, cmap=cm.PRGn)
    plt.savefig("evolution_" + name + '.png')

def multiple_line_plot(X, Y1, Y2, Y3):
    fig = plt.figure()
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2000.0/float(DPI),1500.0/float(DPI))
    df=pd.DataFrame({'x': X, 'data1': Y1, 'data2': Y2, 'data3': Y3 })
    plt.plot( 'x', 'data1', data=df, marker='', color='silver', linewidth=2, label="0%")
    plt.plot( 'x', 'data2', data=df, marker='', color='olive', linewidth=2, label="1%")
    plt.plot( 'x', 'data3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="2.5%")
    plt.legend()
    plt.xscale('log')
    plt.savefig("multiple_line_plot")

def fig_plotting(X, Y, name, three=False, Y2=None, Y3=None):
    fig = plt.figure()
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2000.0/float(DPI),1500.0/float(DPI))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xscale('log')
    #labels = ["$\mathregular{10^1}$","$\mathregular{10^2}$","$\mathregular{10^3}$","$\mathregular{10^4}$","$\mathregular{10^5}$"]
    #plt.xticks([1,10,100,1000,10000,100000], labels)
    plt.plot(X,Y, c='olive')
    plt.savefig("evolution_" + name + '.png')

def plot_fitness_landscape(name):
    X = np.arange(10, 1000)
    Y = np.arange(10, 1000)
    X, Y = np.meshgrid(X, Y)
    Z = Fitness_x_y(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xmin, ymin = np.unravel_index(np.argmin(Z), Z.shape)
    mi = (X[xmin,ymin], Y[xmin,ymin], Z.min())
    #print("The integer minimum value[X,Y,Z]: " + str(mi))
    #print("The integer minimum value[X,Y,Z]: (903, 917, " + str(Fitness_x_y(903,917)) + ")" )
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2500.0/float(DPI),1500.0/float(DPI))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.PRGn)
    plt.savefig("fitness_" + name + '.png')

def printf(string, opt=False, text=""):
    #Print to file
    file_name = "GA_active_datafile" + datetime.datetime.now().strftime("%H")
    if opt:
        file_name += "_key_figs" + text
    result_f = open(file_name, "a")
    result_f.write("\n" + string)
    result_f.close()

if __name__ == '__main__':
    print("------ Genetic Algorithms ------")
    val_av_korning = False
    val = input("Press '1' to plot all figures (takes approx. 15 minutes): ")
    if val == "1":
        val_av_korning = True
    
    population_size = 100 #Should be 100
    generations = 1000 #Not mentioned
    k_Tournament = 5 #Number of Individuals to be used in each tournament
    Pc = 0.6 # Should be tested with 0.6 and without
    bool_crossover = True #Using Crossover
    if bool_crossover == False:
        Pc = 0.0
    Pm = 0.1 #Should be tested with 0.01 and 0.1
    bool_mutation = True #Using Mutation

    evaluate_100_runs = []

    iterations = 1
    successes_sum = 0

    for i in range(iterations):
        print("=====================\nGA run: " + str(i+1)+"\n=====================")
        #(found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr) = Found_On(population)
        temp_hold_metrics = []
        found_optimum_first_on_generation_nr, found_1_pecent_first_on_generation_nr, found_25_percent_first_on_generation_nr, successes = Genetic_Algorithm(population_size, generations, k_Tournament, Pc, bool_crossover, Pm, bool_mutation, val_av_korning)
        printf(
            "\nMETRICS USING: " + "Pop size: " + str(population_size) + ", Pc: " + str(Pc) + ", Pm: " + str(Pm) +
            "\nOptimum found on generation number: " + 
            str(found_optimum_first_on_generation_nr) +
            "\nCumulative Empirical Probability, 1%: " +
            str(found_1_pecent_first_on_generation_nr) +
            "\nCumulative Empirical Probability, 2.5%: " +
            str(found_25_percent_first_on_generation_nr)
        , True, "_")

        print("\nOptimum found on generation number: " + 
            str(found_optimum_first_on_generation_nr) +
            "\nCumulative Empirical Probability, 1%: " +
            str(found_1_pecent_first_on_generation_nr) +
            "\nCumulative Empirical Probability, 2.5%: " +
            str(found_25_percent_first_on_generation_nr)
            )
        successes_sum += successes

        temp_hold_metrics.append(found_optimum_first_on_generation_nr)
        temp_hold_metrics.append(found_1_pecent_first_on_generation_nr)
        temp_hold_metrics.append(found_25_percent_first_on_generation_nr)
        evaluate_100_runs.append(temp_hold_metrics)
        
    X = np.arange(generations*population_size)# list(range(1, generations*population_size+1))
    list_to_plot_optimum = [0] * generations*population_size
    list_to_plot_one_percentage = [0] * generations*population_size
    list_to_plot_twohalf_percentage = [0] * generations*population_size

    for i in range(len(evaluate_100_runs)):
        if evaluate_100_runs[i][0] != 0:
            for ii in range(evaluate_100_runs[i][0], generations*population_size):
                list_to_plot_optimum[ii] += 1
        if evaluate_100_runs[i][1] != 0:
            for ii in range(evaluate_100_runs[i][1], generations*population_size):
                list_to_plot_one_percentage[ii] += 1
        if evaluate_100_runs[i][2] != 0:
            for ii in range(evaluate_100_runs[i][2], generations*population_size):
                list_to_plot_twohalf_percentage[ii] += 1

    printf(str(list_to_plot_optimum), True, "optimum")
    printf(str(list_to_plot_one_percentage), True, "one_percent")
    printf(str(list_to_plot_twohalf_percentage), True, "two_half_percent")

    fig_plotting(X, list_to_plot_optimum, "optimum.png")
    fig_plotting(X, list_to_plot_one_percentage, "one_percent.png")
    fig_plotting(X, list_to_plot_twohalf_percentage, "two_half_percent.png")

    #multiple_line_plot(X, Y1, Y2, Y3):
    Y1 = np.asarray(list_to_plot_optimum)
    Y2 = np.asarray(list_to_plot_one_percentage)
    Y3 = np.asarray(list_to_plot_twohalf_percentage)
    """
    Y1 = np.divide(Y1,iterations)
    Y2 = np.divide(Y2,iterations)
    Y3 = np.divide(Y3,iterations)
    """
    multiple_line_plot(X, Y1, Y2, Y3)
    
    print("Successes: " + str(successes_sum))

    if False: #val_av_korning
        plot_fitness_landscape("test_1")
    print("------ END ------")
    exit()
