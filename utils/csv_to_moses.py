import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', help='path to the CSV file')
parser.add_argument('src_file', help='path to the source file')
parser.add_argument('tgt_file', help='path to the target file')
args = parser.parse_args()

with open(args.csv_file, newline='') as csvfile, \
     open(args.src_file, 'w', newline='') as srcfile, \
     open(args.tgt_file, 'w', newline='') as tgtfile:
    reader = csv.reader(csvfile)
    for row in reader:
        srcfile.write(row[0] + '\n')
        tgtfile.write(row[1] + '\n')