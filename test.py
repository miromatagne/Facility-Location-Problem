import flp

print(flp.solve_flp("FLP-100-20-0.txt", False))

obj, x, y = flp.solve_flp("FLP-100-20-0.txt", True)

print(obj, x, y)

obj, x, y = flp.initial_solution_flp("FLP-100-20-0.txt")

print(obj, x, y)

print(flp.local_search_flp(x, y))
