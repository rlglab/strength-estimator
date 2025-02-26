#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt


def read_data(input_file_name):
    with open(input_file_name, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip().split('\t')  # Get headers, ignoring the first cell
    rows = [line.strip().split('\t') for line in lines[1:]]

    x_values = [int(row[0]) for row in rows]
    y_values = {header[i]: [float(row[i + 1]) for row in rows] for i in range(len(header))}

    return x_values, y_values


def plot_and_save(x_values, y_values, output_file_name='acc.png'):
    for key, values in y_values.items():
        plt.plot(x_values, values, label=key)

    plt.xlabel('Game Used')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(output_file_name, dpi=300)


if len(sys.argv) == 2:
    input_file_name = sys.argv[1]
    output_file_name = 'acc.png'
elif len(sys.argv) == 3:
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
else:
    print("plot.py input_file output_file")
    exit(0)

x_values, y_values = read_data(input_file_name)
plot_and_save(x_values, y_values, output_file_name)
