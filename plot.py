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


if __name__ == "__main__":
    int_solution_time_i()
    int_solution_time_i_over_j()
