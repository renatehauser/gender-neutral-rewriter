import argparse
import csv
import gzip
import io
import json
import logging
import time
import itertools

import spacy
from nltk.corpus import stopwords
from pathlib import Path
from multiprocessing import Pool, cpu_count

import zstandard
from spacy.matcher import PhraseMatcher

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("logfile.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

SPACY_MODEL = spacy.load("de_core_news_sm", disable=["morphologizer", "parser", "ner"])
SPACY_MODEL.enable_pipe("senter")
DE_STOPWORDS = stopwords.words("german")


class Terminology:

    FIELDNAMES = [
        "type",
        "term",
        "alternative",
        "singular_masculine",
        "singular_feminine",
        "plural_masculine",
        "plural_feminine",
        "singular_gender_neutral",
        "plural_gender_neutral",
    ]

    def __init__(self, terminology_file):
        self._current_id = -1
        self.terms = []
        self._terms_by_string = {}
        self._read_terminology_from_file(terminology_file)
        self.terms_by_id = {term.id: term for term in self.terms}

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _read_terminology_from_file(self, terminology_file):
        with open(terminology_file, "r", encoding="utf-8") as term_file:
            terminology = csv.DictReader(
                term_file, delimiter=";", fieldnames=Terminology.FIELDNAMES
            )
            next(terminology)
            for row in terminology:
                # only match neutral forms, not pair or star forms
                # TODO: check if pair and star forms really should be skipped
                if not row["type"] == "neut":
                    continue
                # iterate over terms in a row
                entry = {}
                for field in Terminology.FIELDNAMES[3:]:
                    # check if field is empty
                    if not row[field]:
                        entry[field] = None
                        continue
                    term_string = row[field]
                    gender = self._get_gender_from_fieldname(field)
                    number = self._get_number_from_fieldname(field)
                    neut_type = row["type"]
                    # check if term already exists in terminology
                    if not self._term_exists(term_string, gender, number, neut_type):
                        # create a termID and create term entry
                        term_id = self._get_new_id()
                        term = Term(term_id, term_string, gender, number, neut_type)
                        self.terms.append(term)
                        # to keep track of already existing terms
                        if self._terms_by_string.get(term.term):
                            self._terms_by_string[term.term].append(term)
                        else:
                            self._terms_by_string[term.term] = [term]
                    else:
                        term = self._get_term(term_string, gender, number, neut_type)

                    entry[field] = term

                # update terms with correspondences
                self._update_correspondences(entry)

    @staticmethod
    def _update_correspondences(entry):
        if entry.get("singular_masculine"):
            entry["singular_masculine"].add_correspondence(entry.get("singular_gender_neutral"))
        if entry.get("singular_feminine"):
            entry["singular_feminine"].add_correspondence(entry.get("singular_gender_neutral"))
        if entry.get("singular_gender_neutral"):
            entry["singular_gender_neutral"].add_correspondence(entry.get("singular_masculine"))
        if entry.get("singular_gender_neutral"):
            entry["singular_gender_neutral"].add_correspondence(entry.get("singular_feminine"))
        if entry.get("plural_masculine"):
            entry["plural_masculine"].add_correspondence(entry.get("plural_gender_neutral"))
        if entry.get("plural_feminine"):
            entry["plural_feminine"].add_correspondence(entry.get("plural_gender_neutral"))
        if entry.get("plural_gender_neutral"):
            entry["plural_gender_neutral"].add_correspondence(entry.get("plural_masculine"))
        if entry.get("plural_gender_neutral"):
            entry["plural_gender_neutral"].add_correspondence(entry.get("plural_feminine"))

    def _term_exists(self, term, gender, number, type):
        existing_terms = self._terms_by_string.get(term)
        if not existing_terms:
            return False
        for t in existing_terms:
            if t.gender == gender and t.number == number and t.type == type:
                return True
        return False

    def _get_term(self, term, gender, number, type):
        existing_terms = self._terms_by_string.get(term)
        if not existing_terms:
            return None
        for t in existing_terms:
            if t.gender == gender and t.number == number and t.type == type:
                return t
        return None

    def _get_new_id(self):
        self._current_id += 1
        return self._current_id

    def _get_number_from_fieldname(self, fieldname):
        if fieldname[0:8] == "singular":
            return "SG"
        else:
            return "PL"

    def _get_gender_from_fieldname(self, fieldname):
        if fieldname[-7:] == "neutral":
            return "neut"
        elif fieldname[-8:] == "feminine":
            return "f"
        else:
            return "m"

    @property
    def counts(self):
        counts = {term.id: term.count for term in self.terms}
        return counts

    @property
    def gendered_terms(self):
        gendered_terms = {
            term.id: term for term in self.terms if term.gender == "m" or term.gender == "f"
        }
        return gendered_terms

    @property
    def masculine_terms(self):
        m_terms = {
            term.id: term for term in self.terms if term.gender == "m"
        }
        return m_terms

    @property
    def feminine_terms(self):
        f_terms = {
            term.id: term for term in self.terms if term.gender == "f"
        }
        return f_terms

    @property
    def neutral_terms(self):
        neutral_terms = {term.id: term for term in self.terms if term.gender == "neut"}
        return neutral_terms

    @property
    def gendered_terms_with_correspondences(self):
        # TODO: find out suitable format
        ...

    @property
    def neutral_terms_with_correspondences(self):
        # TODO: find out suitable format
        ...

    def update_count(self, index, count):
        term = self.terms_by_id.get(index)
        if term:
            term.count += count

    def write_counts(self, count_outpath):
        with open(count_outpath, "w", encoding="utf-8", newline="") as outf:
            writer = csv.writer(outf, delimiter=",")
            writer.writerow(
                [
                    "type",
                    "number",
                    "gender",
                    "term",
                    "count",
                    "correspondences",
                ]
            )
            for term in self.terms:
                correspondences = ";".join([self.terms_by_id[i].term for i in term.correspondences])
                row = [term.type, term.number, term.gender, term.term, term.count, correspondences]
                writer.writerow(row)


class Term:
    def __init__(self, id, term, gender, number, type, count=0):
        self.id = id
        self.term = term
        self.gender = gender
        self.number = number
        self.type = type
        self.count = count
        self.correspondences = []
        self.f_correspondences = []
        self.m_correspondences = []

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return self.term

    def add_correspondence(self, term, gender="all"):
        # term can be None
        if term:
            if gender == "f":
                self.f_correspondences.append(term.id)
            elif gender == "m":
                self.m_correspondences.append(term.id)
            else:
                self.correspondences.append(term.id)

    def increase_count(self, n):
        self.count += n


class TermMatcher:
    def __init__(self, oscar_path, terminology, match_level, unmatched_only=False):
        self.oscar_path = oscar_path
        self.terminology = terminology
        self.match_level = match_level
        self.unmatched_only = unmatched_only
        self.matcher = self._get_matcher(self.terminology.terms_by_id)
        self.neut_matcher = self._get_matcher(self.terminology.neutral_terms)
        self.gendered_matcher = self._get_matcher(self.terminology.gendered_terms)

    def _get_matcher(self, terms):
        # create spacy docs for terms
        # TODO: should stopwords be removed?
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
        # PhraseMatchers can't be serialized, therefore have to be deleted
        del state["matcher"]
        del state["neut_matcher"]
        del state["gendered_matcher"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate PhraseMatchers
        self.matcher = self._get_matcher(self.terminology.terms_by_id)
        self.neut_matcher = self._get_matcher(self.terminology.neutral_terms)
        self.gendered_matcher = self._get_matcher(self.terminology.gendered_terms)

    def _search_oscar_files(self, outpath, inspection=False, nr_cpus=None):
        oscar_files = index_files(self.oscar_path, suffixes=["jsonl", "zst", "gz"])
        num_cpus = nr_cpus or (cpu_count() // 3 * 2)
        logger.info(f"Running on {num_cpus} CPUs")
        logger.info(f"Filtering {len(oscar_files)} files.")
        pool = Pool(processes=(num_cpus))
        for inf in oscar_files:
            logger.info(f"Starting processing {inf}...")
            start = time.time()
            # extract unmatched segments to complement training data
            if self.unmatched_only:
                unmatched_outfile = f"{outpath}/unmatched/seg.unm.extracted.{inf.stem.replace('.jsonl', '')}.txt"

                with gzip.open(inf, mode="rt") as inp, open(unmatched_outfile, mode="w", encoding="utf-8") as unm_outp:
                    try:
                        data = [json.loads(d) for d in inp]
                        if len(data) == 0:
                            continue
                    except:
                        continue

                    # divide data into chunks
                    chunk_size = len(data) // (cpu_count() // 2)
                    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

                    # process chunks in parallel
                    results = pool.map(self._process_chunk, chunks)

                    # aggregate and write results
                    logger.info(f"Writing extracted data to files.")
                    for _, _, _, _, _, unm_segs in results:
                        for seg in unm_segs:
                            seg = seg.replace("\n", " ")
                            unm_outp.write(f"{seg}\n")
            # normal filtering with terminology
            else:
                doc_outfile = f"{outpath}/doc/doc.extracted.{inf.stem}.gz"
                neut_outfile = f"{outpath}/neutral/seg.neut.extracted.{inf.stem.replace('.jsonl', '')}.{'csv' if inspection else 'txt'}"
                gen_outfile = f"{outpath}/gendered/seg.gen.extracted.{inf.stem.replace('.jsonl', '')}.{'csv' if inspection else 'txt'}"
                both_outfile = f"{outpath}/both/seg.both.extracted.{inf.stem.replace('.jsonl', '')}.{'csv' if inspection else 'txt'}"
                with open(inf, mode="rb") as inp, gzip.open(doc_outfile, mode="wb") as doc_outp, open(
                    neut_outfile, mode="w", encoding="utf-8"
                ) as neut_outp, open(gen_outfile, mode="w", encoding="utf-8") as gen_outp, open(
                    both_outfile, mode="w", encoding="utf-8"
                ) as com_outp:
                    if inspection:
                        neut_writer = csv.writer(neut_outp, delimiter=";")
                        gen_writer = csv.writer(gen_outp, delimiter=";")
                        com_writer = csv.writer(com_outp, delimiter=";")
                    dctx = zstandard.ZstdDecompressor()
                    stream_reader = dctx.stream_reader(inp)
                    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
                    try:
                        data = [json.loads(d) for d in text_stream]
                    except:
                        continue
                    # divide data into chunks
                    chunk_size = len(data) // (cpu_count() // 2)
                    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

                    # process chunks in parallel
                    results = pool.map(self._process_chunk, chunks)

                    # aggregate and write results
                    logger.info(f"Writing extracted data to files.")
                    for match_counts, out_data, neut_segs, gen_segs, com_segs, _ in results:
                        for i, count in match_counts.items():
                            self.terminology.update_count(i, count)
                        for doc in out_data:
                            out = f"{json.dumps(doc)}\n".encode("utf-8")
                            doc_outp.write(out)
                        for matches, seg in neut_segs:
                            seg = seg.replace("\n", " ")
                            if inspection:
                                neut_writer.writerow([",".join(matches), seg])
                            else:
                                neut_outp.write(f"{seg}\n")
                        for matches, seg in gen_segs:
                            seg = seg.replace("\n", " ")
                            if inspection:
                                gen_writer.writerow([",".join(matches), seg])
                            else:
                                gen_outp.write(f"{seg}\n")
                        for matches, seg in com_segs:
                            seg = seg.replace("\n", " ")
                            if inspection:
                                com_writer.writerow([",".join(matches), seg])
                            else:
                                com_outp.write(f"{seg}\n")
            del results

            end = time.time()
            logger.info(f"Done after {end - start}s!")
        pool.close()
        pool.join()

    def _process_chunk(self, data):
        match_counts = {i: 0 for i in self.terminology.terms_by_id.keys()}
        out_data, neutral_segs, gendered_segs, common_segs, unmatched_segs = [], [], [], [], []

        for d in data:
            if len(d["content"]) > SPACY_MODEL.max_length:
                SPACY_MODEL.max_length = len(d["content"]) + 100

 
        spacy_docs = SPACY_MODEL.pipe((d["content"] for d in data))

        for oscar_doc, doc in zip(data, spacy_docs):
            txt = oscar_doc["content"]

            matches = self.matcher(doc)
            for match in matches:
                match_id = int(SPACY_MODEL.vocab.strings[match[0]])
                match_counts[match_id] += 1
            if self.unmatched_only:
                    unmatched = self._get_segments_without_matches(doc)
                    unmatched_segs.extend(unmatched)
            if not self.unmatched_only and matches:
                neut_segs = self._get_segments_with_matches(doc, matcher="neutral")
                gen_segs = self._get_segments_with_matches(doc, matcher="gendered")
                com_segs, neut_segs, gen_segs = self._sort_out_common_segments(neut_segs, gen_segs)
                neutral_segs.extend(neut_segs)
                gendered_segs.extend(gen_segs)
                common_segs.extend(com_segs)
                out_data.append(oscar_doc)

        return match_counts, out_data, neutral_segs, gendered_segs, common_segs, unmatched_segs

    def _sort_out_common_segments(self, neut_segs, gen_segs):
        gen_only_segs = [seg[1] for seg in gen_segs]
        common_segs = [seg for seg in neut_segs if seg[1] in gen_only_segs]
        common_only_segs = [seg[1] for seg in common_segs]
        neut_segs = [seg for seg in neut_segs if seg[1] not in common_only_segs]
        gen_segs = [seg for seg in gen_segs if seg[1] not in common_only_segs]
        return common_segs, neut_segs, gen_segs

    def _get_segments_without_matches(self, doc):
        for sent in doc.sents:
            matches = self.matcher(sent)
            if matches:
                continue
            else:
                yield sent.orth_

    def _get_segments_with_matches(self, doc, matcher):
        segs_with_matches = []
        for sent_doc in doc.sents:
            if matcher == "neutral":
                neut_matches = self.neut_matcher(sent_doc)
                if neut_matches:
                    neut_matches = [
                        self.terminology.terms_by_id[int(SPACY_MODEL.vocab.strings[match[0]])].term
                        for match in neut_matches
                    ]
                    segs_with_matches.append((neut_matches, sent_doc.orth_))
            elif matcher == "gendered":
                gen_matches = self.gendered_matcher(sent_doc)
                if gen_matches:
                    gen_matches = [
                        self.terminology.terms_by_id[int(SPACY_MODEL.vocab.strings[match[0]])].term
                        for match in gen_matches
                    ]
                    segs_with_matches.append((gen_matches, sent_doc.orth_))
        return segs_with_matches

    def count_and_extract(self, outpath, inspection=False, nr_cpus=None):
        self._search_oscar_files(outpath, inspection, nr_cpus)


class TermCounter:

    def __init__(self, terminology, match_level):
        self.terminology = terminology
        self.match_level = match_level
        self.matcher = self._get_matcher(self.terminology.terms_by_id)
        self.neut_matcher = self._get_matcher(self.terminology.neutral_terms)
        self.gendered_matcher = self._get_matcher(self.terminology.gendered_terms)

    def _get_matcher(self, terms):
        # create spacy docs for terms
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
        # PhraseMatchers can't be serialized, therefore have to be deleted
        del state["matcher"]
        del state["neut_matcher"]
        del state["gendered_matcher"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate PhraseMatchers
        self.matcher = self._get_matcher(self.terminology.terms_by_id)
        self.neut_matcher = self._get_matcher(self.terminology.neutral_terms)
        self.gendered_matcher = self._get_matcher(self.terminology.gendered_terms)

    @staticmethod
    def _generate_chunks(segs_file, chunk_size):
        while True:
            chunk = list(itertools.islice(segs_file, chunk_size))
            if not chunk:
                break
            else:
                yield chunk

    def _search_file(self, infile, nr_cpus=None, chunk_size=500000):
        num_cpus = cpu_count() // 2
        logger.info(f"Running on {num_cpus} CPUs")

        with Pool(processes=(int(num_cpus))) as pool:
            with open(infile, "r", encoding="utf-8") as segments:
                logger.info(f"Starting replacements in {infile}...")
                start = time.time()

                # create chunks from the spacy docs
                chunks = self._generate_chunks(segments, chunk_size)
                results = pool.map(self._process_chunk, chunks)

                for match_counts in results:
                    for i, count in match_counts.items():
                        self.terminology.update_count(i, count)

                end = time.time()
                logger.info(f"Done after {end - start}s!")

    def _process_chunk(self, data):
        match_counts = {i: 0 for i in self.terminology.terms_by_id.keys()}

        for d in data:
            if len(d) > SPACY_MODEL.max_length:
                SPACY_MODEL.max_length = len(d) + 100

 
        spacy_docs = SPACY_MODEL.pipe((d for d in data))

        for doc in spacy_docs:
            matches = self.matcher(doc)
            for match in matches:
                match_id = int(SPACY_MODEL.vocab.strings[match[0]])
                match_counts[match_id] += 1

        return match_counts


    def count(self, infile, nr_cpus=None):
        self._search_file(infile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", help="Path to OSCAR corpus files")
    parser.add_argument("--terminology", help="Path to terminology csv file")
    parser.add_argument(
        "--extracted", help="Path to directory where extracted OSCAR docs should be stored"
    )
    parser.add_argument("--count", help="Path to outfile containing the term counts")
    parser.add_argument(
        "--match-level",
        choices=["lemma", "orth"],
        default="lemma",
        help="Specify if matcher should match by lemma or orthographically. Defaults to lemma.",
    )
    parser.add_argument(
        "--inspection",
        action="store_true",
        help="Set this option to write segments as csv instead of txt with found matches prefixed to segment",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        help="Set number of CPU cores that should be used. If nothing is set, 2/3 of the number of available cores is used.",
    )
    parser.add_argument(
        "--count-only", 
        action="store_true",
        help="Don't extract segments but only count occurrences of the terms in the terminology.",
    )
    parser.add_argument(
        "--unmatched-only",
        action="store_true",
        help="Only extract segments where no term was matched."
    )
    return parser.parse_args()


def index_files(indir, suffixes=""):
    fnames = []
    suffixes = [suffixes] if isinstance(suffixes, str) else suffixes

    for suffix in suffixes:
        fnames += Path(indir).glob("**/*" + suffix)

    return fnames


def main(args):
    if args.count_only:
        logger.info(f"***** Start counting in {args.inpath} *****")
        start = time.time()

        terminology = Terminology(args.terminology)
        term_counter = TermCounter(terminology, args.match_level.upper())
        term_counter.count(args.inpath)
        terminology.write_counts(args.count)

        end = time.time()
        logger.info(f"***** Finished counting. Time taken: {end - start}s *****")
    else:
        logger.info(f"***** Start filtering of OSCAR files *****")
        start = time.time()

        terminology = Terminology(args.terminology)
        term_matcher = TermMatcher(args.inpath, terminology, args.match_level.upper(), unmatched_only=args.unmatched_only)
        term_matcher.count_and_extract(args.extracted, args.inspection)
        terminology.write_counts(args.count)

        end = time.time()
        logger.info(f"***** Finished filtering. Time taken: {end - start}s *****")


if __name__ == "__main__":
    args = parse_args()
    main(args)
