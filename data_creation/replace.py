import argparse
import csv
import random
import logging
import time
import itertools
import os
import uuid
from pathlib import Path

import spacy
import nltk
from nltk.corpus import stopwords
from spacy.matcher import PhraseMatcher
from multiprocessing import Pool, cpu_count

from tools.frequencies import Terminology

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("replacement.logfile.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

SPACY_MODEL = spacy.load("de_core_news_sm", disable=["morphologizer", "parser", "ner"])
nltk.data.path.append("/srv/scratch3/hauser/gender-neutral")
DE_STOPWORDS = stopwords.words("german")


class Replacer:
    def __init__(self, terminology, outprefix, match_level="lemma", target="gendered"):
        self.terminology = terminology
        self.outfile_prefix = outprefix
        self.match_level = match_level
        self.terms = self.terminology.gendered_terms if target == "neutral" else self.terminology.neutral_terms
        self.target = target
        self.matcher = self._get_matcher(self.terms)

    def _get_matcher(self, terms):
        # create spacy docs for terms
        # remove stopwords that produce too many false positive matches
        terms = {i: term.term for i, term in terms.items() if term.term not in DE_STOPWORDS}
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

    def _get_replacement(self, match):
        match_id = int(SPACY_MODEL.vocab.strings[match[0]])
        matched_term = self.terminology.terms_by_id[match_id]
        if self.target == "feminine":
            correspondences = matched_term.f_correspondences
        elif self.target == "masculine":
            correspondences = matched_term.m_correspondences
        else:
            correspondences = matched_term.correspondences
        if correspondences:
            choice = random.choice(correspondences)
            replacement = self.terminology.terms_by_id[choice].term
        else:
            replacement = matched_term.term
        return replacement

    @staticmethod
    def _generate_chunks(segs_file, chunk_size):
        while True:
            chunk = list(itertools.islice(segs_file, chunk_size))
            if not chunk:
                break
            else:
                yield chunk
    
    @staticmethod
    def _generate_chunks_from_files(files):
        for fname in files:
            with open(fname, "r", encoding="utf-8") as inf:
                yield (fname, inf.readlines())
        

    def _process_chunk(self, chunk):
        # FIXME: this file naming scrambles up the order of the files in the output directory!!!
        #i = str(uuid.uuid4())
        #outfile = os.path.join(self.outfile_prefix, f'output_{i}.gen')
        outfile = os.path.join(self.outfile_prefix, f"replaced.{os.path.basename(chunk[0])}")
        start = time.time()
        print(f"Processing file {outfile}", flush=True)
        with open(outfile, "w", encoding="utf-8") as outf:
            seg_docs = SPACY_MODEL.pipe(chunk[1])
            print(f"Done creating spacy docs for file {outfile}", flush=True)
            for segment in seg_docs:
                replaced_segment, _, _ = self._replace_segment(segment)
                outf.write(replaced_segment)
                outf.flush
        print(f"Done with file {outfile} after {time.time() - start}.", flush=True)
        return outfile

    def replace(self, segments_path, outfile, nr_cpus=None, chunk_size=500000):

        num_cpus = nr_cpus or (cpu_count() // 2)
        logger.info(f"Running on {num_cpus} CPUs")
        
        segments_files = index_files(segments_path)

        with Pool(processes=num_cpus) as pool:
            #with open(segments_file, "r", encoding="utf-8") as segments:
            logger.info(f"Starting replacements in {segments_path}...")
            start = time.time()

            # create chunks from the spacy docs
            # chunks = self._generate_chunks(segments, chunk_size)
            chunks = self._generate_chunks_from_files(segments_files)
            results = pool.map(self._process_chunk, chunks)

            end = time.time()
            logger.info(f"Done after {end - start}s!")

    def _replace_segment(self, seg_doc):
        # for reproducibility
        random.seed(1234)

        matches = self.matcher(seg_doc)
        # randomly choose one term if the same token is matched by multiple terms
        matches = self._single_out_matches(matches)
        # get replacement for each match
        replacements = [self._get_replacement(match) for match in matches]

        replaced_segment = ""
        current_index, start = 0, 0
        for match, replacement in zip(matches, replacements):
            _, start, _ = match
            # get text before first match
            if start > current_index:
                replaced_segment += (
                    seg_doc[current_index:start].text + seg_doc[start - 1].whitespace_
                )
            # insert replacement at index of match and add whitespace if available
            replaced_segment += replacement + seg_doc[start].whitespace_
            current_index = start + 1
        # add rest of the segment after last match
        if current_index < len(seg_doc):
            replaced_segment += seg_doc[current_index:].text
        matches = [
            self.terminology.terms_by_id[int(SPACY_MODEL.vocab.strings[match[0]])].term
            for match in matches
        ]
        return replaced_segment, matches, replacements

    @staticmethod
    def _single_out_matches(matches):
        per_start_index = {}
        for match in matches:
            if per_start_index.get(match[1]):
                per_start_index[match[1]].append(match)
            else:
                per_start_index[match[1]] = [match]
        singled_out_matches = []
        for i, candidates in per_start_index.items():
            if len(candidates) > 1:
                singled_out_matches.append(random.choice(candidates))
            else:
                singled_out_matches.append(candidates[0])
        return singled_out_matches


def index_files(indir, suffixes=""):
    fnames = []
    suffixes = [suffixes] if isinstance(suffixes, str) else suffixes

    for suffix in suffixes:
        fnames += Path(indir).glob("**/*" + suffix)
    fnames  = [fname for fname in fnames if os.path.isfile(fname)]
    return fnames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminology", help="Path to terminology csv file")
    parser.add_argument(
        "--segments",
        help="Path to directory with files containing the segments where neutral matches should be replaced",
    )
    parser.add_argument("--match-level", choices=["orth", "lemma"], default="lemma")
    parser.add_argument(
        "--outprefix",
        help="Path to file where replaced segments should be written to !without file extension!",
    )
    parser.add_argument(
        "--inspection",
        action="store_true",
        help="Set this option to write segments as csv instead of txt with replaced matches prefixed to segment",
    )
    parser.add_argument(
        "--target",
        choices=["masculine", "feminine", "neutral", "gendered"],
        default="gendered",
        help="Set the direction for the replacement.",
    )
    parser.add_argument(
        "--cores",
        type=int
    )
    return parser.parse_args()


def main(args):
    terminology = Terminology(args.terminology)
    logger.info(f"Terminology is loaded.")
    replacer = Replacer(terminology, args.outprefix, match_level=args.match_level, target=args.target)
    logger.info("Replacer is initialized.")
    outpath = f"{args.outprefix}.{'csv' if args.inspection else 'txt'}"
    replacer.replace(args.segments, outpath, nr_cpus=args.cores)


if __name__ == "__main__":
    args = parse_args()
    main(args)
