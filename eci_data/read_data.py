#!/usr/bin/env python
import argparse
import json
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sentences')
    ap.add_argument('labels', nargs='?')

    args = ap.parse_args()

    sentence_data = open(args.sentences, 'r')
    
    if args.labels:
        label_data = open(args.labels, 'r')
        for sentence, label in zip(it_sentences(sentence_data), it_labels(label_data)):
            # Tenemos la oración en sentence con su categoría en label
            print(sentence, '\t', label)
            #pass
    else:
        #for sentence, pairID in zip(it_sentences(sentence_data), it_pairID(sentence_data)):
        #    # Tenemos una oración en sentence
        #    print(sentence, '\t', pairID)
        for t in it_tuple(sentence_data):
            print(t[0], '\t', t[1])
        #for t in it_sentencePairID(sentence_data):
        #    print(t[0], '\t', t[1])

    

def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['sentence2']

def it_pairID(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['pairID']

def it_tuple(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield (example['sentence2'], example['gold_label'])

def it_sentencePairID(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield (example['sentence2'], example['pairID'])

def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']


#example['pairID']

main()

#max 55
#min 2
#aver 8.353078642552326
