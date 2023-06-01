import argparse
import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", help="Path to directory with csvs of results")
    parser.add_argument("--output-path", help="Path to save aggregated results to")
    parser.add_argument("--plot", help="Path where plot should be saved")
    parser.add_argument("--frequency-analysis", action="store_true", help="Whether to plot frequency analysis")
    parser.add_argument("--header", nargs="*", help="Header to use for aggregated results")
    return parser.parse_args()


def plot_frequency_anlaysis(results, header, plot_path):
    """
    plot results that have are in rows of the form: [modelname, gender, accuracy bucket 0, accuracy bucket 1, ...]
    """
    # Get the column names for the buckets (assuming they are consistent across rows)
    bucket_columns = header[2:]

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Iterate over each row in the data
    for row in results:
        # Get the model name and gender from the row
        model_name = row[0]
        gender = row[1]

        # Get the bucket values from the row
        bucket_values = [float(value) for value in row[2:]]

        # Plot the bucket values as a colored line
        ax.plot(bucket_values, label=f'{model_name}')

    x_ticks = [0, 1, 11, 101, 1001, 10001]
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks)

    # Set the y-axis limits
    ax.set_ylim(0, 1)

    # Set the y-axis tick locations and labels
    y_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(y_ticks)

    # Add gridlines for the y-axis ticks
    ax.grid(axis='y', linestyle='--')

    # Add a title to the plot
    ax.set_title(f'Frequency Analysis of {"Feminine" if gender == "f" else "Masculine"} Terms')

    # Add a legend and labels
    ax.legend()
    ax.set_xlabel('Frequency Bucket')
    ax.set_ylabel('Terminology / Gender-Fair Match Accuracy')

    # Save the plot to a file
    plt.savefig(plot_path)

    # Close the figure to free up resources
    plt.close(fig)


def main(args):
    results = []
    for file in os.listdir(args.results_path):
        if file.endswith(".csv"):
            with open(os.path.join(args.results_path, file), "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                results.extend(rows[1:])
    with open(args.output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(args.header)
        for line in results:
            writer.writerow(line)
    if args.frequency_analysis and args.plot:
        plot_frequency_anlaysis(results, args.header, args.plot)


if __name__ == "__main__":
    args = parse_args()
    main(args)
