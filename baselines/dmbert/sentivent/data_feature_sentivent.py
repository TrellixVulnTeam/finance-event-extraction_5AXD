# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:32:42 2021

@author: David Xu
"""

import json
import os


def shuffle_docs(input_file, shuffle_output_dir="shuffle_out", test_split_proportion=0.05):
    """
    shuffles docs and output train, dev and test set json file

    :param input_file: the path of input file
    :param shuffle_output_dir: output dir. If not, will be created automatically
    :param test_split_proportion: the proportion of test and dev set.
    """
    import random
    with open(input_file, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines = list(map(eval, lines))  # transform str to dict list

    if not os.path.exists(shuffle_output_dir):
        os.makedirs(shuffle_output_dir)

    train_split = int(len(lines) * (1 - 2 * test_split_proportion))
    dev_split = int(len(lines) * (1 - 1 * test_split_proportion))
    with open("{}/train.json".format(shuffle_output_dir), "w") as f:
        json.dump(lines[:train_split], f)
    with open("{}/dev.json".format(shuffle_output_dir), "w") as f:
        json.dump(lines[train_split:dev_split], f)
    with open("{}/test.json".format(shuffle_output_dir), "w") as f:
        json.dump(lines[dev_split:], f)


def tokens_type_none(sentence, dataset, empty_event, labels):
    """
    A sentence is meaningless and all tokens' trigger_type will be None

    :param sentence: all tokens
    :param dataset: store the dict including tokens, event_type, trigger_tokens, trigger_start and trigger_end
    :param empty_event: count the sentence without any trigger
    :param labels: count the frequency of labels
    :return: dataset, empty_event, labels
    """
    for w_idx in range(len(sentence)):
        empty_event += 1
        data = {"tokens": sentence,
                "event_type": "None",
                "trigger_tokens": [sentence[w_idx]],
                "trigger_start": w_idx,
                "trigger_end": w_idx}
        dataset.append(data)
        labels["None"] += 1
    return dataset, empty_event, labels


def tokens_type_labels(sentence, dataset, events, labels, offset):
    """
    A sentence is meaningful and some tokens' trigger_type will be labeled

    :param sentence: all tokens
    :param dataset: store the dict including tokens, event_type, trigger_tokens, trigger_start and trigger_end
    :param events: event_type list from raw dataset
    :param labels: count the frequency of labels
    :param offset: index the trigger_start in a sentence instead of the index in a doc
    :return: dataset, labels
    """
    # Record the index of triggers in a sentence
    event_ids = [event[0][0] - offset for event in events]
    # define the tokens' trigger_type
    for w_idx in range(len(sentence)):
        if w_idx in event_ids:
            data = {"tokens": sentence,
                    # change / to - for the same ACE format
                    "event_type": events[0][0][1].replace("/", "-"),
                    "trigger_tokens": [sentence[w_idx]],
                    "trigger_start": w_idx,
                    "trigger_end": w_idx}
            dataset.append(data)
            if data["event_type"] not in labels:
                labels[data["event_type"]] = 1
            else:
                labels[data["event_type"]] += 1
    # define the non-trigger tokens
    for w_idx in range(len(sentence)):
        if w_idx not in event_ids:
            data = {"tokens": sentence,
                    "event_type": "None",
                    "trigger_tokens": [sentence[w_idx]],
                    "trigger_start": w_idx,
                    "trigger_end": w_idx}
            dataset.append(data)
            labels["None"] += 1
    return dataset, labels


def main():
    input_file = "./raw_data/all.jsonl"
    shuffle_output_dir = "./shuffle_out/"
    output_dir = "./shuffles_full_out/"
    sets = ["train", "test", "dev"]
    shuffle_docs(input_file=input_file, shuffle_output_dir=shuffle_output_dir, test_split_proportion=0.05)
    for k, s in enumerate(sets):
        # Load shuffled dataset
        with open("{}{}.json".format(shuffle_output_dir, s), "r") as f:
            strings = f.readlines()[0]

        # Transform strings to dict list
        docs = eval(strings)
        # Count the frequency of every label
        labels = {"None": 0}
        # Count the sentence without any trigger
        empty_event = 0
        # Count the total amount of sentences of a set
        total = 0
        # Store the dict including tokens, event_type, trigger_tokens, trigger_start and trigger_end
        dataset = []
        for i, doc in enumerate(docs):
            # The structure of a doc is below:
            # {"doc_key": str,
            #  "sentences": list[list],
            #  "events": list[list],
            #  "ner": list,
            #  "clusters": list,
            #  "relations": list}
            sentences = doc["sentences"]
            events = doc["events"]
            # index the trigger_start in a sentence instead of the index in a doc
            offset = 0
            # data featuring
            for j, sentence in enumerate(sentences):
                if not events[j]:
                    # If the event for a sentence is [], its event_type will be 'None' for every token
                    dataset, empty_event, labels = tokens_type_none(sentence, dataset, empty_event, labels)
                else:
                    # If the event for a sentence is not [], its event_type will be 'None' or a label for every token
                    dataset, labels = tokens_type_labels(sentence, dataset, events[j], labels, offset)
                offset += len(sentence)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total += len(dataset)
        # output
        with open("{}{}.json".format(output_dir, s), "w") as f:
            json.dump(dataset, f)

        print("The frequency of labels in {}".format(s), labels)
        print("The amount of sentences in {}".format(s), total)
        print("The amount of empty-trigger sentences in {}".format(s), empty_event)
        print()


if __name__ == "__main__":
    main()
