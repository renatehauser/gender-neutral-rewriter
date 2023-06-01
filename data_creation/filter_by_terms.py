import argparse
import spacy
import logging
import time
import itertools
from multiprocessing import Pool, cpu_count
from tools.frequencies import Terminology, Term
from spacy.matcher import PhraseMatcher


SPACY_MODEL = spacy.load("de_core_news_sm", disable=["morphologizer", "parser", "ner"])



# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("termfiltering.logfile.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Filter:
    def __init__(self, terminology, match_level="lemma", target="all"):
        self.terminology = terminology
        self.match_level = match_level
        if target == "m":
            self.terms = self.terminology.masculine_terms
        elif target == "f":
            self.terms = self.terminology.feminine_terms
        else:
            self.terms = self.terminology.terms_by_id
        self.matcher = self._get_matcher(self.terms)

    def _get_matcher(self, terms):
        # create spacy docs for terms
        # TODO: should stopwords be removed?
        terms = {i: term.term for i, term in terms.items()}
        prep_terms = list(SPACY_MODEL.pipe(terms.values()))
        # initialize PhraseMatcher
        matcher = PhraseMatcher(prep_terms[0].vocab, attr=self.match_level)
        for term_id, term in zip(terms.keys(), prep_terms):
            match_id = str(term_id)
            matcher.add(match_id, [term])

        return matcher

    # needed for serialization for multiprocessing
    def __getstate__(self):
        state = self.__dict__.copy()
        # PhraseMatcher can't be serialized, therefore has to be deleted
        del state["matcher"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate PhraseMatcher
        self.matcher = self._get_matcher(self.terms)


    @staticmethod
    def _generate_chunks(src_segs_file, trg_segs_file, chunk_size):
        while True:
            chunk = list(itertools.islice(zip(src_segs_file, trg_segs_file), chunk_size))
            if not chunk:
                break
            else:
                yield chunk

    
    def filter(self, src_segments_file, trg_segments_file, outprefix, nr_cpus=None, chunk_size=500000):

        num_cpus = nr_cpus or (cpu_count() // 2)
        logger.info(f"Running on {num_cpus} CPUs")

        src_out = f"{outprefix}.filtered.src"
        trg_out = f"{outprefix}.filtered.trg"
        removed = 0
        with Pool(processes=num_cpus) as pool:
            with open(src_segments_file, "r", encoding="utf-8") as src_segments, open(trg_segments_file, "r", encoding="utf-8") as trg_segments, open(src_out, "w", encoding="utf-8") as src_outf,  open(trg_out, "w", encoding="utf-8") as trg_outf:
                logger.info(f"Starting filtering {src_segments_file}...")
                start = time.time()

                # create chunks from the spacy docs
                chunks = self._generate_chunks(src_segments, trg_segments, chunk_size)
                results = pool.map(self._process_chunk, chunks)

                for filtered_segs, n_filtered_out in results:
                    removed += n_filtered_out
                    for src_seg, trg_seg in filtered_segs:
                        src_outf.write(src_seg)
                        trg_outf.write(trg_seg)

                end = time.time()
                logger.info(f"Done after {end - start}s! {removed} segments were removed")


    def _process_chunk(self, chunk):
        src_segs = (pair[0] for pair in chunk)
        trg_segs = (pair[1] for pair in chunk)
        src_seg_docs = SPACY_MODEL.pipe(src_segs)
        trg_seg_docs = SPACY_MODEL.pipe(trg_segs)
        filtered_segs = [(src_seg.text, trg_seg.text) for src_seg, trg_seg in zip(src_seg_docs, trg_seg_docs) if self._keep_segment(src_seg)]
        filtered_out = len(chunk) - len(filtered_segs)
        return filtered_segs, filtered_out


    def _keep_segment(self, segment):
        matches = self.matcher(segment)
        if matches:
            return True
        else:
            return False



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminology", help="Path to terminology csv file")
    parser.add_argument(
        "--src-segments",
        help="Path to file containing the segments that should be filtered",
    )
    parser.add_argument("--trg-segments")
    parser.add_argument("--match-level", choices=["orth", "lemma"], default="lemma")
    parser.add_argument(
        "--outprefix",
        help="Path to file where replaced segments should be written to !without file extension!",
    )
    parser.add_argument(
        "--target",
        choices=["m", "f", "all"],
        default="all",
        help="Set if segments should be filtered for only M-terms, F-terms or for all terms.",
    )
    parser.add_argument(
        "--cores",
        type=int
    )
    return parser.parse_args()


def main(args):
    terminology = Terminology(args.terminology)
    logger.info(f"Terminology is loaded.")
    f = Filter(terminology, match_level=args.match_level, target=args.target)
    logger.info("Filter is initialized.")
    f.filter(args.src_segments, args.trg_segments, args.outprefix, nr_cpus=5, chunk_size=1000)


if __name__ == "__main__":
    args = parse_args()
    main(args)

