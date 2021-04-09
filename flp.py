import pyomo.environ as pyo
import time
import sys


def read_instance(file_name):
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
    opening_cost, demand, capacity, travel_cost = read_instance(instance_name)
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
    opening_cost, demand, capacity, travel_cost = read_instance(instance_name)
    # print(travel_cost)
    xbar = [[0 for j in range(len(capacity))] for i in range(len(demand))]
    ybar = [0 for j in range(len(capacity))]

    # TODO: check if ok because solve_flp re-reads the file again (waste)
    _, x_star, y_star = solve_flp(instance_name, True)

    j_values = range(len(capacity))
    print(y_star)
    # temp = sorted(zip(y_star, j_values), reverse=True)
    # j_values = [x for _, x in temp]
    # y_star = [y for y, _ in temp]
    j_values = sorted(range(len(capacity)),
                      key=lambda k: y_star[k], reverse=True)

    print(j_values)

    for j_prime in range(len(capacity)):
        j = j_values[j_prime]
        ybar[j] = 1
        # temp = sorted(zip([x_star[i][j]
        #                    for i in range(len(x_star))], i_values), reverse=True)
        # i_values = [x for _, x in temp]
        # x_star_temp = [y for y, _ in temp]
        i_values = sorted(range(len(demand)),
                          key=lambda k: x_star[k][j], reverse=True)

        for i_prime in range(len(demand)):
            i = i_values[i_prime]
            capacity_constraint = sum(xbar[k][j] for k in range(len(xbar)))
            demand_constraint = sum(xbar[i][l] for l in range(len(xbar[i])))

            # if ((i == 34 or i == 46 or i == 91 or i == 7 or i == 2) and j == 0):
            #     print(i, j)
            #     print("Capacity constraint :", capacity_constraint)
            #     print("Demand constraint :", demand_constraint)
            #     print("Demand :", demand[i])
            #     print("Capacity :", capacity[j])
            #     print("xij :", xbar[i][j])
            if capacity_constraint < capacity[j] and demand_constraint < demand[i]:
                xbar[i][j] = min(capacity[j] - capacity_constraint,
                                 demand[i] - demand_constraint)
                if ((i == 34 or i == 46 or i == 91 or i == 7 or i == 2) and j == 0):
                    print("new xij", xbar[i][j])

        if all([sum([xbar[i][j] for j in range(len(capacity))]) >= demand[i] for i in range(len(demand))]):
            travel_cost_matrix = [
                [0 for j in range(len(capacity))] for i in range(len(demand))]
            for key in travel_cost:
                travel_cost_matrix[key[0]][key[1]] = travel_cost[key]
            # print(travel_cost_matrix)
            sum_y = sum(i[0] * i[1]
                        for i in zip(ybar, list(opening_cost.values())))
            sum_x = sum(sum(a * b for a, b in zip(*rows))
                        for rows in zip(xbar, travel_cost_matrix))
            obj = sum_x + sum_y
            return (obj, xbar, ybar)
    return None


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == "__main__":
    # print(read_instance("FLP-100-20-0.txt"))
    # if len(sys.argv) != 3:
    #     print("Usage: flp.py <filename> <solving option>")
    #     exit(1)
    # obj, x, y = solve_flp(sys.argv[1], sys.argv[2] == "--lp")
    # print(y)
    obj, x, y = initial_solution_flp("FLP-100-20-0.txt")
    for j in range(len(x[0])):
        for i in range(len(x)):
            if x[i][j] != 0:
                print("x_" + str(i) + "_" + str(j) + " = " + str(x[i][j]))
    print(obj)
