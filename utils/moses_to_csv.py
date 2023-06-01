import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', help='path to the CSV file')
parser.add_argument('src_file', help='path to the source file')
parser.add_argument('tgt_file', help='path to the target file')
args = parser.parse_args()


with open(args.src_file, 'r') as src, open(args.tgt_file, 'r') as trg, open(args.csv_file, 'w', newline='') as csv_out:
    writer = csv.writer(csv_out, delimiter=',')
    for src_line, trg_line in zip(src, trg):
        writer.writerow([src_line.strip(), trg_line.strip()])

