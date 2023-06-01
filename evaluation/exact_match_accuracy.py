import argparse
import csv
import os
from collections import defaultdict
from tools.frequencies import Terminology

import spacy
from spacy.matcher import PhraseMatcher


SPACY_MODEL = spacy.load("de_core_news_sm", disable=["ner"])


def get_matcher(terms):
    # create spacy docs for terms
    terms = {i: term.term for i, term in terms.items()}
    prep_terms = list(SPACY_MODEL.pipe(terms.values()))
    # initialize PhraseMatcher
    matcher = PhraseMatcher(prep_terms[0].vocab, attr="lemma")
    for term_id, term in zip(terms.keys(), prep_terms):
        match_id = str(term_id)
        matcher.add(match_id, [term])
    return matcher


def fuzzy_match_accuracy(source_file: str, target_file: str, terminology: Terminology, matcher: PhraseMatcher, number=None):
    """
    Calculate the fuzzy match accuracy of a source and target file with a terminology.
    """
    with open(source_file, 'r', encoding='utf8') as src_file, open(target_file, 'r', encoding='utf8') as trg_file:
        print(f"Evaluating {source_file} and {target_file}.")
        src_lines = src_file.readlines()
        trg_lines = trg_file.readlines()
        print(f"Len src: {len(src_lines)}")
        print(f"Len trg: {len(trg_lines)}")
        assert len(src_lines) == len(trg_lines)

        src_line_docs = SPACY_MODEL.pipe(src_lines)
        trg_line_docs = SPACY_MODEL.pipe(trg_lines)

        correct_matches = 0
        incorrect_matches = 0
        for src_line, trg_line in zip(src_line_docs, trg_line_docs):
            src_matches = matcher(src_line)
            # group the matches by their spans
            match_dict = defaultdict(list)
            for match_id, start, end in src_matches:
                match_dict[(start, end)].append(match_id)

            # iterate over each matched span and check if a corresponding term is found in the target
            target_matched_spans = []
            for span_matches in match_dict.values():  # src_matches:
                found = False
                for src_match in span_matches:
                    src_match_id = int(SPACY_MODEL.vocab.strings[src_match])
                    matched_term = terminology.terms_by_id[src_match_id]
                    # only take the src_match into account if it has the correct number (if number is specified)
                    if number and matched_term.number != number:
                        continue
                    correspondences = {i: terminology.terms_by_id[i] for i in matched_term.correspondences}
                    if not correspondences:
                        continue
                    correspondences_matcher = get_matcher(correspondences)
                    trg_matches = correspondences_matcher(trg_line)
                    for i, trg_match in enumerate(trg_matches):
                        # if a target term has already been matched, this doesn't count as a correct match.
                        # this assumes that there are no crossing src-trg-matches
                        if trg_match[1:] in target_matched_spans:
                            continue
                        correct_matches += 1
                        found = True
                        target_matched_spans.append((trg_matches[i][1], trg_matches[i][2]))
                        # only the first matching target term is taken into account
                        break
                    # only one of the possible source matches has to have a correspondence in the target
                    if found:
                        break
                if not found:
                    incorrect_matches += 1

        accuracy = correct_matches / (correct_matches + incorrect_matches)
        print(f"Correct matches: {correct_matches}, Total source terms: {correct_matches + incorrect_matches}, Accuracy: {accuracy}")
        return accuracy


def gf_match_accuracy(source_file: str, target_file: str, src_matcher: PhraseMatcher):
    with open(source_file, 'r', encoding='utf8') as src_file, open(target_file, 'r', encoding='utf8') as trg_file:
        print(f"Evaluating {source_file} and {target_file}.")
        src_lines = src_file.readlines()
        trg_lines = trg_file.readlines()
        print(f"Len src: {len(src_lines)}")
        print(f"Len trg: {len(trg_lines)}")
        assert len(src_lines) == len(trg_lines)

        src_line_docs = SPACY_MODEL.pipe(src_lines)
        trg_line_docs = SPACY_MODEL.pipe(trg_lines)

        correct_matches, missed_matches, additional = 0, 0, 0
        additional_reformulations = []
        for src_line, trg_line in zip(src_line_docs, trg_line_docs):
            src_matches = src_matcher(src_line)
            # group the matches by their spans
            match_dict = defaultdict(list)
            for match_id, start, end in src_matches:
                match_dict[(start, end)].append(match_id)
            src_line_tokens = [token for token in src_line]
            matched_lemmas = []
            for start, end in match_dict.keys():
                matched_lemmas.extend([token.lemma_ for token in src_line[start:end]])
            found_gfm_nouns = 0
            for token in trg_line:
                if "@@GFM@@in" in token.text or "@@GFM@@innen" in token.text:
                    # check if the lemma of the token without the @@GFM@@-suffix is in the matched source term lemmas
                    modified_text = token.text.replace("@@GFM@@innen", "").replace("@@GFM@@in", "")
                    m_modified_text = token.text.replace("@@GFM@@innen", "en").replace("@@GFM@@in", "e")
                    f_modified_text = token.text.replace("@@GFM@@innen", "innen").replace("@@GFM@@in", "in")
                    lemma, m_lemma, f_lemma = SPACY_MODEL(modified_text)[0].lemma_, SPACY_MODEL(m_modified_text)[0].lemma_, SPACY_MODEL(f_modified_text)[0].lemma_
                    if m_lemma in matched_lemmas or f_lemma in matched_lemmas or lemma in matched_lemmas:
                        found_gfm_nouns += 1
                    else:
                        additional += 1
                        additional_reformulations.append((token.text, src_line.text))
            to_be_matched = len(match_dict.keys())
            # count only as many gfm nouns as there are source matches
            if found_gfm_nouns > to_be_matched:
                found_gfm_nouns = to_be_matched

            missed_matches += to_be_matched - found_gfm_nouns
            correct_matches += found_gfm_nouns

        accuracy = correct_matches / (correct_matches + missed_matches)
        additional_proportion = additional / (correct_matches + missed_matches) * 100
        print(f"Correct matches: {correct_matches}, Total source terms: {correct_matches + missed_matches}, Accuracy: {accuracy}")
        print(f"Additional: {additional}, {additional_proportion}% more unmatched nouns were gender-fairly reformulated")
        return accuracy, additional_proportion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--sort-files", action="store_true")
    parser.add_argument("--terminology-path", required=True)
    parser.add_argument("--number")
    parser.add_argument("--gender")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gn", action="store_true")
    parser.add_argument("--gf", action="store_true")
    parser.add_argument("--count-additional", action="store_true")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--header", nargs="*", default=['model', 'gender', '0', '1', '11', '101', '1001', '10001'])
    return parser.parse_args()


def main(args):
    terminology = Terminology(args.terminology_path)
    matcher = get_matcher(terminology.gendered_terms)

    src_files = [f for f in os.listdir(args.source_path) if os.path.isfile(os.path.join(args.source_path, f))]
    trg_files = [f for f in os.listdir(args.target_path) if os.path.isfile(os.path.join(args.target_path, f))]
    if args.sort_files:
        src_files = sorted(src_files, key=lambda x: int(os.path.splitext(x)[0]))
        trg_files = sorted(trg_files, key=lambda x: int(os.path.splitext(x)[0]))

    print(f"Len src_files: {len(src_files)}", flush=True)
    print(f"Len trg_files: {len(trg_files)}", flush=True)
    assert len(src_files) == len(trg_files), "Source and target path must contain the same number of files."
    row = [args.model]
    if args.gender:
        row.append(args.gender)
    # split the target path into directories
    for src, trg in zip(src_files, trg_files):
        src_path = os.path.join(args.source_path, src)
        trg_path = os.path.join(args.target_path, trg)
        if args.gf:
            print(f"Computing gf Match Accuracy.")
            acc, additional = gf_match_accuracy(src_path, trg_path, matcher)
        # Computing Lemma Match Accuracy is the default, if nothing is set
        else:
            print(f"Computing Lemma Match Accuracy.")
            acc = fuzzy_match_accuracy(src_path, trg_path, terminology, matcher, args.number)
        
        row.append(acc)
        if args.gf and args.count_additional:
            row.append(additional)

    with open(args.out_file, 'w', encoding='utf8') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(args.header)
        writer.writerow(row)


if __name__ == "__main__":
    args = parse_args()
    main(args)
