import pyomo.environ as pyo


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
    model.I = pyo.RangeSet(0, len(demand)-1)
    model.J = pyo.RangeSet(0, len(capacity) - 1)

    # Params
    model.f = pyo.Param(model.J, initialize=opening_cost, default=0)
    model.c = pyo.Param(model.J, initialize=capacity, default=0)
    model.d = pyo.Param(model.I, initialize=demand, default=0)
    model.t = pyo.Param(model.I, model.J, initialize=travel_cost, default=0)

    # Variables
    model.x = pyo.Var(model.I, model.J, domain=pyo.PositiveIntegers)
    model.y = pyo.Var(model.J, domain=pyo.Binary)

    # Objective function
    model.obj = pyo.Objective(rule=obj_expression)

    # Constraints
    model.constraint2 = pyo.Constraint(model.J, rule=constraint_rule_2)
    model.constraint3 = pyo.Constraint(model.I, rule=constraint_rule_3)

    # Optimizer
    opt = pyo.SolverFactory('glpk')
    opt.solve(model, tee=True)
    print(pyo.value(model.obj))

    # return (obj,x,y)


def initial_solution_flp(instance_name):
    pass
    # return (obj,x,y)


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


# print(read_instance("FLP-100-20-0.txt"))
solve_flp("FLP-100-20-0.txt", True)
