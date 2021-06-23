import pandas as pd
from math import pow, sqrt
import random
import time


def launch(alpha, early_stop):
    """
    The frame of the whole program.
    Apply the GRASP algorithm by iterating the search for max_iteration times. Update the solution and corresponding
    cost if a better solution is found.
    :param alpha: the level of greediness
    :return: alpha for statistical purpose best_cost is the best cost of the founded best solution
    """

    input_tsp_file = read_data()
    max_iteration = 1000
    early_stop = early_stop
    greedy_factor = alpha
    start = time.time()

    best_cost = float('inf')
    best_sol = None

    while max_iteration > 0:
        print('ITERATION %d' % max_iteration)
        max_iteration -= 1
        new_sol, new_cost = construct_greedy_solution(input_tsp_file, greedy_factor)
        new_sol, new_cost = local_search(new_sol, new_cost, early_stop)

        if new_cost < best_cost:
            best_cost = new_cost
            best_sol = new_sol
            print('New solution found:\nCost:%.2f' % best_cost)

    stop = time.time()

    running_time = stop - start

    print('Best cost: %.2f, Elapsed: %.2f' % (best_cost, running_time))
    print('Best solution: %s' % best_sol)
    return alpha, best_cost


def construct_greedy_solution(nodes, alpha):
    """
    Construct a greedy solution.
    The key idea is create a intermediate list to store the cost of the relatively nearer node w.r.t the last node
    in solution list. Every time after adding a node from RCL into solution list, calculate the the distance between
    each left node and this node. Then put the nearer nodes into the RCL list. The level of greediness is controlled
    by alpha.
    :param nodes: the node list of a problem i.e. berlin52.
    :param alpha: the parameter to control the level of greediness. 0 is completely greedy and 1 is not greedy at all.
    :return: a new solution and corresponding total cost.
    """

    solution = []
    problem_size = len(nodes)
    solution.append(nodes[random.randrange(0, problem_size)])  # randomly add the first node into solution

    #  keep adding nodes into solution until the number of nodes in solution equals to problem size
    while len(solution) < problem_size:
        cost = []
        nodes_not_in_solution = [node for node in nodes if node not in solution]

        for node in nodes_not_in_solution:
            cost.append(distance(solution[-1], node))

        max_cost, min_cost = max(cost), min(cost)
        rcl = []

        for index, cost in enumerate(cost):
            if cost <= min_cost + alpha*(max_cost-min_cost):
                rcl.append(nodes_not_in_solution[index])

        selected_node = rcl[random.randrange(0, len(rcl))]
        solution.append(selected_node)

    new_cost = tour_cost(solution)

    return solution, new_cost


def local_search(sol, cost, early_stop):
    """
    A local search to randomly explore the possible neighbors of a solution.
    :param sol: a solution
    :param cost: corresponding cost of a solution
    :param early_stop: local search will be stopped if there is no increase after early_stop runs
    :return: a new solution and corresponding cost
    """

    count = 0

    while count < early_stop:
        new_sol = stochastic_swap(sol)  # randomly swap two edges to explore the possible neighbors.
        new_cost = tour_cost(new_sol)  # calculate the total cost of a solution

        #  update the solution and cost if a better solution is found
        if new_cost < cost:
            sol = new_sol
            cost = new_cost
            count = 0
        else:
            count += 1

    return sol, cost


def tour_cost(sol):
    """
    Calculate the euclidean distance of a solution by sum the distance of every distance between each two nodes
    :param sol: a solution made up of a series of nodes
    :return: total distance of a route
    """

    total_distance = 0

    for index in range(len(sol)):
        start_node = sol[index]
        if index == len(sol) - 1:
            end_node = sol[0]
        else:
            end_node = sol[index + 1]

        total_distance += distance(start_node, end_node)

    return total_distance


def distance(node_1, node_2):
    """
    Calculate the euclidean distance between two nodes.
    :param node_1: a node
    :param node_2: another node
    :return: distance between two nodes
    """

    distance_two_nodes = 0

    for x, y in zip(node_1, node_2):
        distance_two_nodes += sqrt(pow((x-y), 2))
    return distance_two_nodes


def stochastic_swap(sol):
    """
    Swap two edges in the route.
    Randomly select two nodes which are different and non-consecutive. Swap the edges between these two nodes
    then reverse the route between them.

    :param sol: a solution made up of a series of nodes
    :return: a new solution
    """

    sol_copy = sol[:]  # create a copy
    sol_size = len(sol)

    node_1_index = random.randrange(0, sol_size)  # randomly select a node
    node_2_index = random.randrange(0, sol_size)  # randomly select another node

    exclude_set = {node_1_index}  # create a forbidden set to guarantee node 2 is not node 1 or the neighbor of node1

    #  the rules exclude set
    if node_1_index == 0:
        exclude_set.add(sol_size-1)
    else:
        exclude_set.add(node_1_index-1)

    if node_1_index == sol_size - 1:
        exclude_set.add(0)
    else:
        exclude_set.add(node_1_index+1)

    #  if the selected node 2 is in the exclude set, select again
    while node_2_index in exclude_set:
        node_2_index = random.randrange(0, sol_size)

    #  to guarantee that node 1 index < node 2 index
    if node_2_index < node_1_index:
        node_1_index, node_2_index = node_2_index, node_1_index

    #  reversed the route between two selected nodes
    sol_copy[node_1_index:node_2_index] = reversed(sol_copy[node_1_index:node_2_index])

    return sol_copy


def read_data():
    """
    Read a txt data file.
    :return: a list of the data file
    """

    f = pd.read_csv('berlin52-tsp.txt', sep='\s+')
    f = pd.DataFrame(f).to_numpy().tolist()
    return f
