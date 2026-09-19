import pyomo.environ as pyo

# --- Problem Data ---
NumNodes = 4
Source = 1
Destination = 4

LinkCost = {(1, 2): 4, (1, 3): 8, (2, 3): 4, (2, 4): 7, (3, 4): 5}
LinkCapacity = {(1, 2): 10, (1, 3): 4, (2, 3): 5, (2, 4): 10, (3, 4): 10}
DemandValue = 1  # Standard flow demand (h) for shortest path

# --- Model Definition ---
model = pyo.ConcreteModel()

# Sets
model.V = pyo.RangeSet(1, NumNodes)
model.E = pyo.Set(dimen=2, initialize=list(LinkCost.keys()))

# Parameters
model.cost = pyo.Param(model.E, initialize=LinkCost)
model.capacity = pyo.Param(model.E, initialize=LinkCapacity)
model.demand = pyo.Param(initialize=DemandValue)

# Decision Variables
# x[i,j] = 1 if edge (i,j) is active; 0 otherwise
model.x = pyo.Var(model.E, within=pyo.Binary, initialize=0)


# Objective Function: Minimize total link cost
def mincost_rule(m):
    return sum(m.cost[i, j] * m.x[i, j] for (i, j) in m.E)


model.mincost = pyo.Objective(rule=mincost_rule, sense=pyo.minimize)


# Flow Conservation Constraints
def flow_rule(m, k):
    inFlow = sum(m.x[j, i] for (j, i) in m.E if i == k)
    if k == Source:
        inFlow += 1

    outFlow = sum(m.x[i, j] for (i, j) in m.E if i == k)
    if k == Destination:
        outFlow += 1

    return inFlow == outFlow


model.flow = pyo.Constraint(model.V, rule=flow_rule)


# Capacity Constraints
def limit_rule(m, i, j):
    return m.x[i, j] * m.demand <= m.capacity[i, j]


model.limit = pyo.Constraint(model.E, rule=limit_rule)

# --- Solve the Model ---
solver = pyo.SolverFactory("glpk")
results = solver.solve(model)

# --- Display Results ---
print(f"Optimal Path Cost: {pyo.value(model.mincost)}")
print("Active Links along the shortest path:")
for i, j in model.E:
    if pyo.value(model.x[i, j]) > 0.5:
        print(
            f"  Link ({i} -> {j}) with cost {model.cost[i, j]} and capacity {model.capacity[i, j]}"
        )
