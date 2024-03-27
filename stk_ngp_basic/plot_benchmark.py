import pandas as pd
import matplotlib.pyplot as plt

# The data should look like this:
# Name, NumElementsX, NumElementsY, NumElementsZ, NumNodes, GPU, Time
# DirectFunction, 10,10,100,12221,1,1.16348e-05
# StandardFunction, 10,10,100,12221,1,9.69048e-05
# ...

def make_plot(file_base):
    # Load the data, first row is the header
    data_cpu = pd.read_csv('build/'+file_base+'.csv')
    data_cpu.columns = data_cpu.columns.str.strip()

    data_gpu = pd.read_csv('build_gpu/'+file_base+'.csv')
    data_gpu.columns = data_gpu.columns.str.strip()

    # Plot the data
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for name, group in data_cpu.groupby('Name'):
        color = colors.pop(0)
        # Skip name = "StkForEachEntity"
        if name == "StkForEachEntity":
            continue
        label = name + " CPU"
        if name == "StkForEachEntityAbstracted":
            label = "StkForEachEntity CPU"
        plt.loglog(group['NumNodes'], group['Time'], label=label, marker='x', linestyle=':', color=color)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for name, group in data_gpu.groupby('Name'):
        color = colors.pop(0)
        # Skip name = "DirectFunction", "LambdaFunction", "StandardFunction", "StkForEachEntity" for GPU
        if name == "DirectFunction" or name == "LambdaFunction" or name == "StandardFunction" or name == "StkForEachEntity":
            continue
        label = name + " GPU"
        if name == "StkForEachEntityAbstracted":
            label = "StkForEachEntity GPU"
        plt.loglog(group['NumNodes'], group['Time'], label=label, marker='o', linestyle='--', color=color)

    plt.legend()
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time Per Iteration (s)')
    plt.grid()
    plt.savefig(file_base+'.png')

if __name__ == '__main__':
    file_base = 'benchmark_orig'
    make_plot(file_base)