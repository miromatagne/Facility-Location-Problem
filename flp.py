import random

import pyomo.environ as pyo
import time
import copy


class Instance:
    def __init__(self, file_name):
        # Store all the attributes of the instance
        self.opening_cost, self.demand, self.capacity, self.travel_cost = read_instance(
            file_name)

        # Compute the travel cost matric of the instance
        self.travel_cost_matrix = travel_cost_to_matrix(
            len(self.capacity), len(self.demand), self.travel_cost)


def read_instance(file_name):
    """
        Reads the problem instance and extracts all the usefuk information from the
        instance file.

        :param file_name: name of the instance file to be read
        :return: all the information relative to that instance
    """
    opening_cost = {}
    demand = {}
    capacity = {}
    travel_cost = {}
    try:
        file = open("Instances/{}".format(file_name), 'r')
        info = file.readline().split(" ")
        I = int(info[0])
        J = int(info[1])
        info = file.readline().split(" ")
        for j in range(J):
            opening_cost[j] = int(info[j])
        info = file.readline().split(" ")
        for i in range(I):
            demand[i] = int(info[i])
        info = file.readline().split(" ")
        for j in range(J):
            capacity[j] = int(info[j])
        for i in range(I):
            info = file.readline().split(" ")
            for j in range(J):
                travel_cost[(i, j)] = int(info[j])
    except:
        print("Error reading file.")
    return opening_cost, demand, capacity, travel_cost


def obj_expression(m):
    return pyo.summation(m.f, m.y) + pyo.summation(m.t, m.x)


def constraint_rule_2(m, j):
    return sum(m.x[i, j] for i in m.I) <= m.c[j] * m.y[j]


def constraint_rule_3(m, i):
    return sum(m.x[i, j] for j in m.J) >= m.d[i]


def solve_flp(instance_name, linear):
    """
        Solve an FLP instance using the GLPK solver. The solution can either
        be an integer solution or an LP relaxation.

        :param instance_name: name of the FLP instance to solve
        :param linear: True if we wish to solve the LP relaxation
        :return: (obj,x,y) where obj is the objective function corresponding
                 to the solutions x and y
    """
    # opening_cost, demand, capacity, travel_cost = read_instance(instance_name)
    opening_cost, demand, capacity, travel_cost = instance.opening_cost, instance.demand, instance.capacity, instance.travel_cost
    model = pyo.ConcreteModel()

    # Sets
    model.I = pyo.RangeSet(0, len(demand) - 1)
    model.J = pyo.RangeSet(0, len(capacity) - 1)

    # Params
    model.f = pyo.Param(model.J, initialize=opening_cost, default=0)
    model.c = pyo.Param(model.J, initialize=capacity, default=0)
    model.d = pyo.Param(model.I, initialize=demand, default=0)
    model.t = pyo.Param(model.I, model.J, initialize=travel_cost, default=0)

    # Variables
    if linear:
        model.x = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals)
        model.y = pyo.Var(model.J, domain=pyo.NonNegativeReals, bounds=(0, 1))
    else:
        model.x = pyo.Var(model.I, model.J, domain=pyo.NonNegativeIntegers)
        model.y = pyo.Var(model.J, domain=pyo.Binary)

    # Objective function
    model.obj = pyo.Objective(rule=obj_expression)

    # Constraints
    model.constraint2 = pyo.Constraint(model.J, rule=constraint_rule_2)
    model.constraint3 = pyo.Constraint(model.I, rule=constraint_rule_3)

    # Optimizer
    TIME_LIMIT = 600
    opt = pyo.SolverFactory('glpk')
    opt.options['tmlim'] = TIME_LIMIT
    start = time.time()
    # reset timer as problem has been solved
    results = opt.solve(model, tee=True, timelimit=TIME_LIMIT)
    end = time.time()
    obj = pyo.value(model.obj)
    print(f"Resolution time: {end - start}")
    print(f"Final solution: {obj}")

    x = []
    for i in range(len(demand)):
        sublist = list(model.x[i, :].value)
        x.append(sublist)

    y = list(model.y[:].value)

    return (obj, x, y)


def initial_solution_flp(instance_name):
    """
        Computes an initial feasible integer solution to the FLP instance
        using a greedy algorithm.

        :param instance_name: name of the instance file
        :return: (obj,x,y) where obj is the objective function corresponding
                 to the solutions x and y
    """
    opening_cost, demand, capacity, travel_cost = instance.opening_cost, instance.demand, instance.capacity, instance.travel_cost_matrix

    # Initialize xbar and ybar to 0's
    xbar = [[0 for j in range(len(capacity))] for i in range(len(demand))]
    ybar = [0 for j in range(len(capacity))]

    # TODO: check if ok because solve_flp re-reads the file again (waste)
    # Solve the LP Relaxation of the FLP instance, x_star and y_star are the optimal solutions
    _, x_star, y_star = solve_flp(instance_name, True)

    # Indexes of the sorted y_star elements in descending order
    j_values = sorted(range(len(capacity)),
                      key=lambda k: y_star[k], reverse=True)

    for j_prime in range(len(capacity)):
        # j is the index of the j-th highest y_star
        j = j_values[j_prime]
        ybar[j] = 1

        # Indexes of the sorted x_star elements for a fixed j in descending order
        i_values = sorted(range(len(demand)),
                          key=lambda k: x_star[k][j], reverse=True)

        for i_prime in range(len(demand)):
            i = i_values[i_prime]

            # Sums corresponding to both conditions to be verified
            capacity_constraint = sum(xbar[k][j] for k in range(len(xbar)))
            demand_constraint = sum(xbar[i][l] for l in range(len(xbar[i])))

            # If there is sufficient capacity available and demand is unfulfilled, update
            # xbar[i][j] in order to tighten one of the 2 conditions
            if capacity_constraint < capacity[j] and demand_constraint < demand[i]:
                xbar[i][j] = min(capacity[j] - capacity_constraint,
                                 demand[i] - demand_constraint)

        # If all demands are met, compute the objective value and return the solution
        if all([sum([xbar[i][j] for j in range(len(capacity))]) >= demand[i] for i in range(len(demand))]):
            # Compute both operands of the objective function
            obj = compute_obj_value(xbar, ybar)
            return (obj, xbar, ybar)

    # Return None if no feasible integer solution exists
    return None


def compute_obj_value(x, y):
    """
        Computes the objective value of a solution (x,y)

        :param x: values of the xij variables
        :param y: values of the yj variables
        :return: value of the objective function
    """
    travel_cost_matrix, opening_cost = instance.travel_cost_matrix, instance.opening_cost
    sum_y = sum(i[0] * i[1]
                for i in zip(y, list(opening_cost.values())))
    sum_x = sum(sum(a * b for a, b in zip(*rows))
                for rows in zip(x, travel_cost_matrix))
    obj = sum_x + sum_y
    return obj


def travel_cost_to_matrix(nb_facilities, nb_clients, travel_cost_dict):
    """
        Converts the travel cost dictionnary into a matrix, for more efficient computations
        of the objective function.

        :param nb_facilities: number of facilities
        :param nb_clients: number of clients
        :param travel_cost_dict: dictionnary containing the travel costs
        :return: the 2D matric corresponding to the travel costs
    """
    travel_cost_matrix = [
        [0 for j in range(nb_facilities)] for i in range(nb_clients)]
    for key in travel_cost_dict:
        travel_cost_matrix[key[0]][key[1]] = travel_cost_dict[key]
    return travel_cost_matrix


def local_search_flp(x, y):
    demand, capacity, travel_cost, opening_cost = instance.demand, instance.capacity, instance.travel_cost, instance.opening_cost
    travel_cost_matrix = travel_cost_to_matrix(
        len(capacity), len(demand), travel_cost)
    start_time = time.time()
    tabu_count = 0
    xbar, ybar = copy.deepcopy(x), y.copy()
    #print(check_validity(xbar, ybar))
    best_x, best_y = copy.deepcopy(x.copy()), y.copy()
    best_obj = compute_obj_value(xbar, ybar)
    past_results = []
    non_tabu_count = 0
    while time.time() - start_time < 20:
        r = random.random()
        if r < 0.8:
            xbar, ybar = facility_movement(
                xbar.copy(), ybar.copy(), travel_cost_matrix)
        else:
            xbar, ybar = assignment_movement(xbar.copy(), ybar.copy())
        if (xbar, ybar) not in past_results:
            past_results.append((xbar, ybar))
            obj_bar = compute_obj_value(xbar, ybar)
            if obj_bar < best_obj:
                best_x = copy.deepcopy(xbar)
                #best_x = xbar[:][:]
                best_y = ybar.copy()
                best_obj = obj_bar
                #print("VERIF", check_validity(best_x, best_y))
            non_tabu_count += 1
        else:
            tabu_count += 1
    print("Tabu :", tabu_count)
    print("Non tabu :", non_tabu_count)
    # print(past_results)
    return best_obj, best_x, best_y


def assignment_movement(x, y):
    xbar, ybar = x[:], y[:]
    demand, capacity = instance.demand, instance.capacity
    nb_customers = random.randint(1, 2)
    customers = random.sample(range(len(demand)), nb_customers)
    chosen_demands = {}
    for i in customers:
        chosen_demands[i] = []
        used_facilities = [k for k in range(len(x[i])) if x[i][k] > 0]
        nb_facilities = random.randint(1, min(len(used_facilities), 2))
        facilities = random.sample(used_facilities, nb_facilities)
        # print(x[i])
        # print(len(used_facilities))
        for j in facilities:
            # print(x[i][j])
            chosen_demands[i].append((i, j))
    # print(chosen_demands)

    for c in chosen_demands:
        # print(c)
        if len(chosen_demands[c]) > 0:
            for d in chosen_demands[c]:
                opened_facilities = [i for i in range(len(y)) if y[i] == 1]
                if d[1] in opened_facilities:
                    opened_facilities.remove(d[1])
                remaining_demand = x[d[0]][d[1]]
                while remaining_demand != 0 and len(opened_facilities) > 0:
                    #print("remaining_demand", remaining_demand)
                    new_facility = random.sample(opened_facilities, 1)
                    new_facility = new_facility[0]
                    # print(new_facility)
                    capacity_constraint = sum(
                        xbar[k][new_facility] for k in range(len(xbar)))
                    #print("capacity_constraint", capacity_constraint)
                    #print("capacity", capacity[new_facility])
                    if capacity_constraint < capacity[new_facility]:
                        # print(xbar[d[0]][new_facility])
                        # print(xbar[d[0]][d[1]])
                        amount_realloc = min(
                            capacity[new_facility] - capacity_constraint, remaining_demand)
                        #print("REALLOC", amount_realloc)
                        xbar[d[0]][new_facility] += amount_realloc
                        xbar[d[0]][d[1]] -= amount_realloc
                        # print(xbar[d[0]][new_facility])
                        # print(xbar[d[0]][d[1]])
                        remaining_demand -= amount_realloc
                    else:
                        opened_facilities.remove(new_facility)
    return xbar, ybar


def facility_movement(x, y, travel_cost):
    demand, capacity = instance.demand, instance.capacity
    openable_facilities = [i for i in range(len(y)) if y[i] == 0]
    closable_facilities = [i for i in range(len(y)) if y[i] == 1]
    if len(openable_facilities) == 0:
        # we can not open any facility
        return None
    while True:
        # choose randomly the number of facilities we open or close
        opened_facilities_count = random.randint(
            1, min(len(openable_facilities), 2))
        closed_facilities_count = random.randint(
            1, min(len(closable_facilities), 2))
        # list of indices of facilities that open/close
        opened_facilities = random.sample(
            openable_facilities, opened_facilities_count)
        closed_facilities = random.sample(
            closable_facilities, closed_facilities_count)
        amount_sum = 0
        for j_minus in closed_facilities:
            amount_sum += sum(x[k][j_minus] for k in range(len(x)))
        capacity_sum = 0
        for j_plus in opened_facilities:
            capacity_sum += capacity[j_plus]
        if amount_sum <= capacity_sum:
            break
    # greedy reassign
    xbar, ybar = x[:], y[:]
    for j_plus in opened_facilities:
        ybar[j_plus] = 1
    for j_minus in closed_facilities:
        ybar[j_minus] = 0
        for i in range(len(x)):
            xbar[i][j_minus] = 0
    i_values = sorted(range(len(demand)),
                      key=lambda k: demand[k], reverse=True)
    for i_prime in range(len(i_values)):
        i = i_values[i_prime]
        j_values = sorted(
            range(len(travel_cost[0])), key=lambda k: travel_cost[i][k])
        for j_prime in range(len(j_values)):
            j = j_values[j_prime]
            # Sums corresponding to both conditions to be verified
            capacity_constraint = sum(xbar[k][j] for k in range(len(xbar)))
            demand_constraint = sum(xbar[i][l] for l in range(len(xbar[i])))

            # If there is sufficient capacity available and demand is unfulfilled, update
            # xbar[i][j] in order to tighten one of the 2 conditions
            if ybar[j] == 1 and capacity_constraint < capacity[j] and demand_constraint < demand[i]:
                xbar[i][j] = min(capacity[j] - capacity_constraint,
                                 demand[i] - demand_constraint)
    #print(check_validity(xbar, ybar))
    return xbar, ybar


def check_validity(x, y):
    demand, capacity = instance.demand, instance.capacity
    for j in range(len(x[0])):
        capacity_constraint = sum(x[k][j] for k in range(len(x)))
        if capacity_constraint > capacity[j]*y[j]:
            return False
    for i in range(len(x)):
        demand_constraint = sum(x[i][l] for l in range(len(x[i])))
        if demand_constraint < demand[i]:
            return False
    return True


if __name__ == "__main__":
    # print(read_instance("FLP-100-20-0.txt"))
    # if len(sys.argv) != 3:
    #     print("Usage: flp.py <filename> <solving option>")
    #     exit(1)
    # obj, x, y = solve_flp(sys.argv[1], sys.argv[2] == "--lp")
    # print(y)

    # Global variable corresponding to the instance
    instance = Instance("FLP-100-20-0.txt")

    obj, x, y = initial_solution_flp("FLP-100-20-0.txt")
    obj_sol, x_sol, y_sol = local_search_flp(x, y)
    print("Solution :", obj_sol)
    print("Valid :", check_validity(x_sol, y_sol))
