#!/usr/bin/env python

import argparse, os, re, time
from pyrouge import Rouge155
from util import evaluate_rouge
from g_rouge import rouge


def split_sentences(article, sentence_start_tag='<s>', sentence_end_tag='</s>'):
    return re.findall(r'%s (.+?) %s' % (sentence_start_tag, sentence_end_tag), article)

# convenient decorator
def register_to_registry(registry):
    def _register(func):
        registry[func.func_name] = func
        return func
    return _register

baseline_registry = {}
register = register_to_registry(baseline_registry)

# baseline methods
@register
def first_sentence(article, sentence_start_tag='<s>', sentence_end_tag='</s>'):
    ''' use sentence tags to output the first sentence of an article as its summary. '''
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:1]

@register
def first_three_sentences(article, sentence_start_tag='<s>', sentence_end_tag='</s>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:3]

@register
def first_two_sentences(article, sentence_start_tag='<s>', sentence_end_tag='</s>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:2]

@register
def verbatim(article, sentence_start_tag='<s>', sentence_end_tag='</s>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents

@register
def second_sentence(article, sentence_start_tag='<s>', sentence_end_tag='</s>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[1:2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='Path to the tokenized source file. One sample per line with sentence tags.')
    parser.add_argument('-t', '--target', required=True, help='Path to the tokenized target file. One sample per line with sentence tags.')
    parser.add_argument('-m', '--method', default='first_sentence', choices=baseline_registry.keys(), help='Baseline method to use.')
    parser.add_argument('-d', '--delete', action='store_true', help='Delete the temporary files created during evaluation.')
    parser.add_argument('-g', '--google', action='store_true', help='Evaluate with the ROUGE implementation from google/seq2seq.')
    # ROUGE arguments
    parser.add_argument('--no-stemming', action='store_true', help='Turn off stemming in ROUGE.')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='The number of bootstrap samples used in ROUGE.')

    args = parser.parse_args()

    process = baseline_registry[args.method]

    n_source = 0
    references = []
    summaries = []
    with open(args.source, 'rb') as f:
        for i, article in enumerate(f):
            summary = process(article)
            summaries.append(summary)
            n_source += 1

    n_target = 0
    with open(args.target, 'rb') as f:
        for i, article in enumerate(f):
            candidate = split_sentences(article)
            references.append([candidate])
            n_target += 1

    assert n_source == n_target, 'Source and target must have the same number of samples.'

    rouge_args = rouge_args = [
        '-c', 95, # 95% confidence intervals, necessary for the dictionary conversion routine
        '-n', 2, # up to bigram
        '-a',
        '-r', args.n_bootstrap, # the number of bootstrap samples for confidence bounds
    ]

    if not args.no_stemming:
        # add the stemming flag
        rouge_args += ['-m']

    t0 = time.time()
    # evaluate with official ROUGE script v1.5.5
    scores = evaluate_rouge(summaries, references, remove_temp=args.delete, rouge_args=rouge_args)
    dt = time.time() - t0

    print '* method', args.method

    headers = ['rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score', 'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score', 'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score']

    print headers
    for header in headers:
        print scores[header],
    print

    print '* evaluated %i samples, took %gs, averaging %ss/sample' % (n_target, dt, dt * 1. / n_target)

    if args.google:
        # modified ROUGE implementation based on https://github.com/google/seq2seq
        # modified to support multi-sentence summaries
        t0 = time.time()
        g_scores = rouge(summaries, [candidates[0] for candidates in references])
        dt = time.time() - t0

        g_headers = ['rouge_1/p_score', 'rouge_1/r_score', 'rouge_1/f_score', 'rouge_2/p_score', 'rouge_2/r_score', 'rouge_2/f_score', 'rouge_l/p_score', 'rouge_l/r_score', 'rouge_l/f_score']

        print g_headers
        for header in g_headers:
            print g_scores[header],
        print

        print '* evaluated %i samples, took %gs, averaging %ss/sample' % (n_target, dt, dt * 1. / n_target)
