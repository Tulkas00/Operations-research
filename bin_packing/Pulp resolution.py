from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

def bin_packing_with_pulp(items, bin_capacity):
    
    num_items = len(items)
    num_bins = len(items)

    # Define the problem
    problem = LpProblem("BinPackingProblem", LpMinimize)

    # Decision variables
    x = LpVariable.dicts("x", [(i, j) for i in range(num_items) for j in range(num_bins)], 0, 1, LpBinary)
    y = LpVariable.dicts("y", [j for j in range(num_bins)], 0, 1, LpBinary)

    # Objective: Minimize the number of bins used
    problem += lpSum(y[j] for j in range(num_bins)), "MinimizeBins"

    # Constraints
    # 1. Each item is placed in exactly one bin
    for i in range(num_items):
        problem += lpSum(x[i, j] for j in range(num_bins)) == 1, f"ItemAssignment_{i}"

    # 2. Total weight in each bin does not exceed its capacity
    for j in range(num_bins):
        problem += lpSum(x[i, j] * items[i] for i in range(num_items)) <= bin_capacity * y[j], f"BinCapacity_{j}"

    # Solve the problem
    solver = PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    # Extract results
    bin_assignments = [[] for _ in range(num_bins)]
    for i in range(num_items):
        for j in range(num_bins):
            if x[i, j].varValue > 0.5:  # If the variable is 1
                bin_assignments[j].append(items[i])

    # Remove empty bins
    bin_assignments = [b for b in bin_assignments if b]

    return len(bin_assignments), bin_assignments


# Example usage
items = [30, 10, 20, 40, 35, 15, 25, 10, 20, 45, 5, 25, 15, 10, 30, 20, 40, 10]
bin_capacity = 50

# Solve using PuLP
num_bins, bin_contents = bin_packing_with_pulp(items, bin_capacity)

print(f"Minimum number of bins required: {num_bins}")
print("Bin contents:")
for i, contents in enumerate(bin_contents, 1):
    print(f"  Bin {i}: {contents}")
