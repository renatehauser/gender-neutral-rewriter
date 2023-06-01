import argparse
import sentencepiece as spm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--model-prefix")
    parser.add_argument("--vocab-size")
    parser.add_argument("--character-coverage")
    parser.add_argument("--model-type")
    parser.add_argument("--shuffle-input-sentence", type=bool)
    parser.add_argument("--user-defined-symbols", nargs="*")
    return parser.parse_args()


def main(args):
    spm.SentencePieceTrainer.train(
        input=args.input, 
        model_prefix=args.model_prefix, 
        vocab_size=args.vocab_size, 
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        shuffle_input_sentence=args.shuffle_input_sentence,
        user_defined_symbols=args.user_defined_symbols,
    )



if __name__ == "__main__":
    args = parse_args()
    main(args)