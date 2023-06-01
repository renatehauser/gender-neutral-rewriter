import argparse
import torch

from transformers import pipeline
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

MODEL_NAMES = {"ende": "facebook/wmt19-en-de", "deen": "facebook/wmt19-de-en"}


def get_model_tokenizer(src_lang, trg_lang):
    mname = MODEL_NAMES[f"{src_lang}{trg_lang}"]
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    return model, tokenizer


def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def _generate_chunks(segs_file, chunk_size):
        while True:
            chunk = itertools.islice(segs_file, chunk_size)
            try:
                first_elem = next(chunk)
            except StopIteration:
                break
            yield itertools.chain([first_elem], chunk)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--infile", help="Path to line segmented file with segments to be translated"
    )
    parser.add_argument(
        "-o", "--outfile", help="Path to file where translated segments should be written to"
    )
    parser.add_argument("-s", "--source-lang")
    parser.add_argument("-t", "--target-lang")
    parser.add_argument("-d", "--device", type=int, help="GPU to run translation on")
    return parser.parse_args()


def main(args):
    model, tokenizer = get_model_tokenizer(src_lang=args.source_lang, trg_lang=args.target_lang)
    translator = pipeline(task="translation", model=model, tokenizer=tokenizer, device=args.device)

    with open(args.infile, "r", encoding="utf-8") as infile, open(
        args.outfile, "w", encoding="utf-8"
    ) as outfile:
        for batch in chunk(infile.readlines(), 8):
            try:
                decoded = translator(batch, batch_size=8, max_length=400)
            except torch.cuda.OutOfMemoryError:
                decoded = []
                for seg in batch:
                    transl = translator(seg, max_length=400)
                    decoded.append(transl[0])
            for line in decoded:
                outfile.write(line["translation_text"] + "\n")
                outfile.flush()


if __name__ == "__main__":
    args = parse_args()
    main(args)
