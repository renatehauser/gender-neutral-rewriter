import argparse
import csv
from nltk.corpus import stopwords

BUCKETS = {
    0: {"lower": 0, "upper": 0},
    1: {"lower": 1, "upper": 10},
    11: {"lower": 11, "upper": 100},
    101: {"lower": 101, "upper": 1000},
    1001: {"lower": 1001, "upper": 10000},
    10001: {"lower": 10001, "upper": 100000}
}

def create_buckets(infile1, infile2, number, gender):
    buckets = {0: [], 1: [], 11: [], 101: [], 1001: [], 10001: []}
    with open(infile1, "r", encoding="utf-8") as csv_inf1, open(infile2, "r", encoding="utf-8") as csv_inf2:
        fieldnames = [
                    "type",
                    "number",
                    "gender",
                    "term",
                    "count",
                    "correspondences",
                ]
        counts1 = list(csv.DictReader(csv_inf1, delimiter=",", fieldnames=fieldnames))
        counts2 = list(csv.DictReader(csv_inf2, delimiter=",", fieldnames=fieldnames))
        assert len(counts1) == len(counts2), "The two files don't contain the same number of lines. Please provide parallel files."
        for t1, t2 in zip(counts1[1:], counts2[1:]):
            assert t1["term"] == t2["term"] and t1["number"] == t2["number"] and t1["gender"] == t1["gender"], f"This line is not parallel: {t1} != {t2}. Please provide parallel files."
            if t1["number"] != number or t1["gender"] != gender:
                continue
            bucket = get_bucket(t1, t2)
            if not bucket == 999:
                buckets[bucket].append(t1["term"])
    return buckets

def get_bucket(term1, term2):
    for order, b in BUCKETS.items():
        c1, c2 = int(term1["count"]), int(term2["count"])
        if c1 <= b["upper"] and c2 <= b["upper"] and c1 >= b["lower"] and c2 >= b["lower"]:
            return order
    return 999


def write_buckets(buckets, outf_prefix, number, gender):
    filenames = [f"{outf_prefix}.{gender}.{number}.{i}.txt" for i in BUCKETS.keys()]
    for bucket, fname in zip(buckets.values(), filenames):
        with open(fname, "w", newline="", encoding="utf-8") as outf:
            sorted_terms = sorted(list(set(bucket)))
            for term in sorted_terms:
                # I found two patterns that don't allow the star form: -mann/-frau (e.g. Feuerwehrmann/-frau) and -ling (e.g. Lehrling) 
                if is_star_possible(term, number):
                    outf.write(f"{term}\n")


def is_star_possible(term, number):
    #return not (term.endswith("mann") or term.endswith("frau") or term.endswith("ling") or term.endswith("männer") or term.endswith("frauen") or term.endswith("linge"))
    if number == "SG":
        return not (term.endswith("mann") or term.endswith("frau") or term.endswith("ling"))
    if number == "PL":
        return not (term.endswith("männer") or term.endswith("frauen") or term.endswith("linge"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs=2, help="Paths to the two count files that should be compared to create buckets")
    parser.add_argument("--number")
    parser.add_argument("--gender")
    parser.add_argument("--outprefix")
    return parser.parse_args()


def main(args):
    buckets = create_buckets(args.counts[0], args.counts[1], args.number, args.gender)
    write_buckets(buckets, args.outprefix, args.number, args.gender)


if __name__ == "__main__":
    args = parse_args()
    main(args)