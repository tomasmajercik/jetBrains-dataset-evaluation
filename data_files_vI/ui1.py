import copy
import random
import matplotlib.pyplot as plt

best_fitness = 0
best_trajectory = []
best_number_of_steps = 0
best_number_of_treasures_found = 0
best_gameplan = [[0]*7]*7

def find_instructions(memory): #machine
    instructions = []
    index = 0
    max_instructions = 0
    # while len(instructions) < 16 and index < 64:
    while index < 64 and max_instructions < 500:
        max_instructions -=- 1
        byte_value = memory[index]

        byte_format = format(byte_value, '08b')
        instruction_type = int(byte_format[:2], 2)  # first 2 numbers to get the instruction type
        address = int(byte_format[2:], 2)  # get the last 6 numbers to get address refer

        if instruction_type == 0:  # if 00, increment adress
            memory[address] += 1
            if memory[address] > 255:
                memory[address] -= 255
            index += 1
        elif instruction_type == 1:  # if 01, decrement adress
            memory[address] -= 1
            if memory[address] < 0:
                memory[address] += 255
            index += 1
        elif instruction_type == 2:  # if 10, jump on adress
            index = byte_value
        elif instruction_type == 3:  # if 11, append move
            bin_number = bin(memory[index])  # convert number to binary
            ones_count = bin_number.count('1')  # count how many ones are there

            if 0 < ones_count < 3:
                instructions.append("H")
            elif 2 < ones_count < 5:
                instructions.append("D")
            elif 4 < ones_count < 7:
                instructions.append("P")
            elif 6 < ones_count < 9:
                instructions.append("L")

            index += 1
            if index > 60:
                index = random.randint(0,60)

    return instructions
def generate_individual(first_population_fill):
    memory = [0]*64 #64 memory cells with size of 1 byte
    for m in range(first_population_fill):
        memory[m] = random.randint(1,255) #fill with random values to get movement
    return memory

def take_a_walk(trajectory):
    gameplan = [
        ["o", "o", "o", "o", "o", "o", "o"],
        ["o", "o", "o", "o", "P", "o", "o"],
        ["o", "o", "P", "o", "o", "o", "o"],
        ["o", "o", "o", "o", "o", "o", "P"],
        ["o", "P", "o", "o", "o", "o", "o"],
        ["o", "o", "o", "o", "P", "o", "o"],
        ["o", "o", "o", "x", "o", "o", "o"]
    ]

    steps_count = 0
    treasues_found_count = 0
    position_x = 6
    pos_y = 3
    for step in trajectory:
        if treasues_found_count == 5:
            return steps_count, treasues_found_count, gameplan

        steps_count += 1
        if step == "H":
            position_x -= 1
        elif step == "D":
            position_x += 1
        elif step == "P":
            pos_y += 1
        elif step == "L":
            pos_y -= 1

        #check if the move was succesfull and update the map + add +1 step and if found treasure, +1 treasure
        if position_x < 0 or position_x > 6 or pos_y > 6 or pos_y < 0:  # fall out of map
            # return steps_count, -3000, gameplan
            return steps_count, -3, gameplan
        else:  # did not fall
            if gameplan[position_x][pos_y] == "o" or gameplan[position_x][pos_y] == "x":
                gameplan[position_x][pos_y] = "x"
            elif gameplan[position_x][pos_y] == "P":
                treasues_found_count += 1
                gameplan[position_x][pos_y] = "x"


    return steps_count, treasues_found_count, gameplan
def fitness(indiv):
    indiv_copy = indiv[:]
    trajectory = find_instructions(indiv_copy)

    global best_fitness
    global best_trajectory
    global best_number_of_steps
    global best_number_of_treasures_found
    global best_gameplan

    steps_count, treasues_found_count, gameplan = take_a_walk(trajectory)

    if treasues_found_count > 2:
        fitness_score = treasues_found_count + (0.1000 - (steps_count / 500)) #swap so fewer steps has more points
        if best_fitness < fitness_score:#replace the best individual
            best_fitness = fitness_score
            best_trajectory = trajectory[:]
            best_number_of_steps = steps_count
            best_number_of_treasures_found = treasues_found_count
            best_gameplan = gameplan[:]
        return fitness_score
    elif treasues_found_count < 3:
        fitness_score = treasues_found_count + (steps_count/1000) #swap so fewer steps has more points
        if best_fitness < fitness_score: #replace the best individual
            best_fitness = fitness_score
            best_trajectory = trajectory[:]
            best_number_of_steps = steps_count
            best_number_of_treasures_found = treasues_found_count
            best_gameplan = gameplan[:]
        return fitness_score

def tournament_selection(ranked_population, tournament_size):

    #select few random individuals for the turnament
    tournament = random.sample(ranked_population, tournament_size)

    #sort them by fitness
    tournament.sort(key=lambda x: x[0], reverse=True)

    #return the best individual from tournament
    return tournament[0][1] #[0][1] because it is a tuple
def mutate(individual_parent):

    mutation_type = random.choice([0, 1, 2])

    if mutation_type == 0:
        for i in range(2): #change random number in individual
            random_index = random.randint(5, 60)
            random_value = random.randint(1, 254)

            individual_parent[random_index] = random_value
    elif mutation_type == 1: #switch random bit
        random_index = random.randint(0, 63)
        bit_to_flip = random.randint(0, 7)  # Random bit to flip (0 to 7 for an 8-bit)

        individual_parent[random_index] ^= (1 << bit_to_flip)  #use XOR to flip that bit
    elif mutation_type == 2: # swap two random numbers
        index1 = random.randint(0, 63)
        index2 = random.randint(0, 63)

        individual_parent[index1], individual_parent[index2] = individual_parent[index2], individual_parent[index1]

    return individual_parent


def crossover(parent1, parent2):
    crossover_point = random.randint(20, 40) #random number to split the genes of parents

    #create two childrens witch oposite genes
    child = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child, child2

def print_fitness_values_as_graph(fitness_values):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_values, color='blue', label='Fitness')
    plt.xlabel('Generations', fontsize=14)
    plt.ylabel('Fitness Value', fontsize=14)
    plt.title('Fitness Progress Over Generations', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()



##### main()
# genetic algrotithm variables setting
population_size = 100
elitism_count = 2
mutation_rate = 5 #in %
max_no_update = 250 # maximum no improvement generations limit

tournament_size = 3
first_population_fill = 31

# necessary variables
population = []
fitness_values_for_graph = []

#fill first population
for i in range(population_size): #number of individuals in population
    population.append(generate_individual(first_population_fill))

print("working on it...")

break_from_main = False
# for i in range(500): #genetic algorithm (for test purposes)
while not break_from_main:
    ranked = []
    best_fitness_for_generation = -100
    for individual in population: #fitness(individual) to get the fitness score of the individual and also add individual to not lose it
        individual_copy = individual.copy()
        fitness_value = fitness(individual_copy)
        ranked.append((fitness_value, individual))

        if fitness_value > best_fitness_for_generation:
            best_fitness_for_generation = fitness_value


    fitness_values_for_graph.append(best_fitness_for_generation)

    index = len(fitness_values_for_graph) - 1
    exit_from_loop = 0

    # check if to break from the loop or no
    if len(fitness_values_for_graph) > max_no_update:
        index = len(fitness_values_for_graph) - 1
        exit_from_loop = 0

        for check in range(max_no_update):
            if fitness_values_for_graph[index] == fitness_values_for_graph[-1]:
                exit_from_loop += 1
            else:
                exit_from_loop = 0  # Reset if any difference
            index -= 1

            if exit_from_loop == max_no_update:
                print(f"=no improvement in last {max_no_update} generations: exit from loop=")
                break_from_main = True
                break

    #elitism
    ranked.sort(key=lambda x: x[0], reverse=True)
    elites = [copy.deepcopy(ranked[r][1]) for r in range(elitism_count)] #carry over the best individuals to next generation

    new_population = []
    #tournament selection
    # print_gameplan(gameplan)
    while len(new_population) < (population_size - elitism_count):
        #choose parents by turnament selection
        ranked_copy = ranked[:]
        parent1 = tournament_selection(ranked_copy, tournament_size)
        parent2 = tournament_selection(ranked_copy, tournament_size)

        parent1_copy, parent2_copy = parent1[:], parent2[:]
        child, child2 = crossover(parent1_copy, parent2_copy)
        child = mutate(child.copy())
        child2 = mutate(child2.copy())

        #mutate
        if random.random() < (mutation_rate/100): # % chance of mutation
            child = mutate(child.copy())
        if random.random() < (mutation_rate/100): # % chance of mutation
            child2 = mutate(child2.copy())

        #add children to another generation
        new_population.append(child)
        new_population.append(child2)

    #replace the population with the new one
    population = elites.copy() + new_population.copy() #also ensure that we don't exceed population size


print("===========================")
print("===========================")
print(f"The best program ended with fitness value of {best_fitness}")
print(f"It took {best_number_of_steps} steps")
print(f"And found {best_number_of_treasures_found} treasures")
print("The map:")
for row in best_gameplan:
    print(row)
print("The moves:")
print(best_trajectory)

print_fitness_values_as_graph(fitness_values_for_graph)
# print(fitness_values_for_graph) # in case we want to use the data from graph such as for another script that combines multiple graphs (such as those in documentation)

