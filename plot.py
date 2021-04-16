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
            average_i_comp_time[row["File"].split("-")[1]].append(float(row["Execution time"]))
        except KeyError:
            average_i_comp_time[row["File"].split("-")[1]] = [float(row["Execution time"])]
    for i in average_i_comp_time:
        average_i_comp_time[i] = np.mean(average_i_comp_time[i])
    print(average_i_comp_time)
    plt.title("Average computation time for various I values")
    plt.xlabel("Number of clients I")
    plt.ylabel("Average computation time (s)")
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
            average_i_over_j_comp_time[str(int(row_lst[1]) // int(row_lst[2]))].append(float(row["Execution time"]))
        except KeyError:
            average_i_over_j_comp_time[str(int(row_lst[1]) // int(row_lst[2]))] = [float(row["Execution time"])]
    for i in average_i_over_j_comp_time:
        average_i_over_j_comp_time[i] = np.mean(average_i_over_j_comp_time[i])
    print(average_i_over_j_comp_time)
    plt.title("Average computation time for various I/J ratios")
    plt.xlabel("I/J ratio")
    plt.ylabel("Average computation time (s)")
    plt.plot(average_i_over_j_comp_time.keys(), average_i_over_j_comp_time.values())
    plt.show()


def timout_count():
    int_df = pd.read_csv("int_benchmark.csv")
    int_df = int_df[int_df["Execution time"] >= 600]
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
    comp_df = int_df.set_index("File").join(lp_df.set_index("File"), lsuffix="_int", rsuffix="_lp")
    comp_df = comp_df[comp_df["Execution time_int"] < 600]
    comp_df["Integrality gap"] = comp_df["Solution_int"] / comp_df["Solution_lp"]
    print(comp_df.to_string())


if __name__ == "__main__":
    # int_solution_time_i()
    # int_solution_time_i_over_j()
    # print_filtered_lp()
    timout_count()
    timout_count_ij()
