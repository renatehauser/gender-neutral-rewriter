import argparse
import sentencepiece as spm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    return parser.parse_args()


def main(args):
    sp = spm.SentencePieceProcessor(model_file=args.model)
    with open(args.infile, "r", encoding="utf-8") as inf, open(args.outfile, "w", encoding="utf-8") as outf:
        for line in inf:
            encoded = sp.encode(line, out_type=str)
            outf.write(f"{' '.join(encoded)}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)