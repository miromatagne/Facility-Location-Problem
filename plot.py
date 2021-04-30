import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def int_solution_time_i():
    benchmark_df = pd.read_csv("int_benchmark.csv", sep=",")
    average_i_comp_time = dict()
    # filtering out timeouts leads to a lot of missing values that biases the result
    # benchmark_df = benchmark_df[benchmark_df["Execution time"] < 600]  # filter timeout values
    for i, row in benchmark_df.iterrows():
        try:
            average_i_comp_time[row["File"].split(
                "-")[1]].append(float(row["Execution time"]))
        except KeyError:
            average_i_comp_time[row["File"].split(
                "-")[1]] = [float(row["Execution time"])]
    for i in average_i_comp_time:
        average_i_comp_time[i] = 600 - np.mean(average_i_comp_time[i])
    print(average_i_comp_time)
    plt.title("Average deviation from the timeout threshold for various I values")
    plt.xlabel("Number of clients I")
    plt.ylabel("Average deviation from the timeout threshold (s)")
    plt.plot(average_i_comp_time.keys(), average_i_comp_time.values())
    plt.show()


def int_solution_time_i_over_j():
    benchmark_df = pd.read_csv("int_benchmark.csv", sep=",")
    average_i_over_j_comp_time = dict()
    # filtering out timeouts leads to a lot of missing values that biases the result
    # benchmark_df = benchmark_df[benchmark_df["Execution time"] < 600]  # filter timeout values
    for i, row in benchmark_df.iterrows():
        row_lst = row["File"].split("-")
        try:
            average_i_over_j_comp_time[str(
                int(row_lst[1]) // int(row_lst[2]))].append(float(row["Execution time"]))
        except KeyError:
            average_i_over_j_comp_time[str(
                int(row_lst[1]) // int(row_lst[2]))] = [float(row["Execution time"])]
    for i in average_i_over_j_comp_time:
        average_i_over_j_comp_time[i] = 600 - \
            np.mean(average_i_over_j_comp_time[i])
    print(average_i_over_j_comp_time)
    plt.title("Average deviation from the timeout threshold for various I/J ratios")
    plt.xlabel("I/J ratio")
    plt.ylabel("Average deviation from the timeout threshold (s)")
    plt.plot(average_i_over_j_comp_time.keys(),
             average_i_over_j_comp_time.values())
    plt.show()


def timout_count():
    # this function takes the fist time for each I
    int_df = pd.read_csv("int_benchmark.csv")
    # int_df = int_df[int_df["Execution time"] >= 600]
    timeout_count = dict()
    for i, row in int_df.iterrows():
        try:
            timeout_count[row["File"].split("-")[1]] += 1
        except KeyError:
            timeout_count[row["File"].split("-")[1]] = 1
    print(timeout_count)
    plt.title("Timeout count for various I values")
    plt.xlabel("Number of clients I")
    plt.ylabel("Number of timeouts")
    plt.plot(timeout_count.keys(), timeout_count.values())
    plt.show()


def timout_count_ij():
    int_df = pd.read_csv("int_benchmark.csv")
    int_df = int_df[int_df["Execution time"] >= 600]
    timeout_count = dict()
    for i, row in int_df.iterrows():
        row_lst = row["File"].split("-")
        try:
            timeout_count[str(int(row_lst[1]) // int(row_lst[2]))] += 1
        except KeyError:
            timeout_count[str(int(row_lst[1]) // int(row_lst[2]))] = 1
    print(timeout_count)
    plt.title("Timeout count for various I/J ratios")
    plt.xlabel("I/J ratio")
    plt.ylabel("Number of timeouts")
    plt.plot(timeout_count.keys(), timeout_count.values())
    plt.show()


def print_filtered_lp():
    int_df = pd.read_csv("int_benchmark.csv")
    lp_df = pd.read_csv("LP_benchmark.csv")
    comp_df = int_df.set_index("File").join(
        lp_df.set_index("File"), lsuffix="_int", rsuffix="_lp")
    comp_df = comp_df[comp_df["Execution time_int"] < 600]
    comp_df["Integrality gap"] = comp_df["Solution_int"] / \
        comp_df["Solution_lp"]
    print(comp_df.to_string())


def greedy_optimality_gap():
    int_df = pd.read_csv("int_benchmark.csv")
    greedy_sol = pd.read_csv("greedy_solutions.csv")
    comp_df = int_df.set_index("File").join(
        greedy_sol.set_index("File"), lsuffix="_int", rsuffix="_greedy")
    comp_df = comp_df[comp_df["Execution time"] < 600]
    comp_df["Optimality gap"] = (
        comp_df["Solution_greedy"] - comp_df["Solution_int"]) / comp_df["Solution_int"]
    print(comp_df.to_latex())


def plot_best_solution_evolution(*filenames: str):
    plt.title("Evolution of best solution during local search")
    plt.ylabel("Best objective value (over 10000)")
    plt.xlabel("Time (s)")
    for filename in filenames:
        sol_df = pd.read_csv(filename)
        plt.plot(sol_df["Timestamp"], sol_df["ObjValue"] /
                 1e5, label=filename.split(".txt")[0])
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # int_solution_time_i()
    # int_solution_time_i_over_j()
    # print_filtered_lp()
    # timout_count()
    # timout_count_ij()
    # greedy_optimality_gap()
    plot_best_solution_evolution(
        "FLP-250-100-0.txt_best_history.csv", "FLP-250-100-1.txt_best_history.csv")
