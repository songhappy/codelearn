# from
import numpy as np

def show(time_list):
    first_latency = np.mean(time_list)
    average_2n = time_list[1:]
    average_2n.sort()
    average_2n_latency = np.mean(average_2n)
    p90_latency = average_2n[int(len(average_2n) * 0.9)]
    p99_latency = average_2n[int(len(average_2n) * 0.99)]
    print("p90_latency: ", p90_latency)
    print("P99_latency: ", p99_latency)


def quntiles_90_99(time_list):
    first_latency = np.mean(time_list)
    average_2n = time_list[1:]
    average_2n.sort()
    average_2n_latency = np.mean(average_2n)
    p90_latency = average_2n[int(len(average_2n) * 0.9)]
    p99_latency = average_2n[int(len(average_2n) * 0.99)]
    return (p90_latency, p99_latency)