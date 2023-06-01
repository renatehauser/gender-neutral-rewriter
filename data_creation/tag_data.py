import argparse

TAG = "@@F@@ "

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile")
    parser.add_argument("-o", "--outfile")
    return parser.parse_args()


def main(args):
    with open(args.infile, "r", encoding="utf-8") as inf, open(args.outfile, "w", encoding="utf-8") as outf:
        for segment in inf:
            tagged = TAG + segment
            outf.write(tagged)

if __name__ == "__main__":
    args = parse_args()
    main(args)