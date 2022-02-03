import copy
import json

from utils import *
from seqeval.metrics import f1_score, precision_score, recall_score


def generate_gold_labels(folder_name, name, tokenized_folder):
    input_path = folder_name + '/' + name + "/"
    file_names = get_file_names(input_path)

    label_arr = []
    cnt = 0
    sent_cnt = 0
    for file_name in file_names:
        try:
            with open(folder_name + '/' + tokenized_folder + '/' + file_name + '.json', 'r') as json_file:
                json_obj = json.load(json_file)
                start_token_dict = json_obj["start_token_dict"]
                end_token_dict = json_obj["end_token_dict"]
                tokens = json_obj["tokens"]
                offsets = json_obj["offsets"]

            pre = 0
            start_sentence_offsets = []
            arr = []
            for i in range(len(tokens)):
                if tokens[i] == '.' or i == len(tokens) - 1:
                    start_sentence_offsets.append((offsets[pre][0], pre))
                    arr.append(["O" for _ in range(i - pre + 1)])
                    pre = i + 1
                    sent_cnt += 1

            with open(input_path + file_name + '.a2', 'r') as file:
                while line := file.readline().rstrip():
                    if line[0] != "T":
                        continue
                    indicators = line.split("\t")[1].split()
                    event_type = "B-" + indicators[0]
                    start = int(indicators[1])
                    end = int(indicators[2])
                    if str(start) not in start_token_dict:
                        print("NOT FOUND START:", start, file_name)
                    if str(end) not in end_token_dict:
                        print("NOT FOUND END:", end, file_name)

                    sent_index = upper_bound(start_sentence_offsets, start) - 1
                    start_index = start_token_dict[str(start)] - start_sentence_offsets[sent_index][1]
                    end_index = end_token_dict[str(end)] - start_sentence_offsets[sent_index][1]

                    for token_index in range(start_index, end_index + 1):
                        arr[sent_index][token_index] = event_type

            label_arr.extend(arr)
            cnt += 1
        except Exception as e:
            print(e)
            print("ERROR OCCURS!!")
            break

    with open(folder_name + '/' + name + '_gold.json', 'w') as f:
        json.dump(label_arr, f)

    print("Processed", cnt, "files!")


def evaluate(golds, predictions):
    results = {
        "precision": precision_score(golds, predictions),
        "recall": recall_score(golds, predictions),
        "f1": f1_score(golds, predictions)
    }

    print("***** Eval results *****")
    for key in sorted(results.keys()):
        print("  " + key + " = " + str(results[key]))


def evaluate_flat_nested_triggers(folder_name, name, tokenized_folder):
    with open(folder_name + '/' + name + '_gold.json') as json_file:
        golds = json.load(json_file)

    with open(folder_name + '/' + name + '_predictions.json') as json_file:
        predictions = json.load(json_file)

    flat_mark, nested_mark = generate_flat_nested_mark(folder_name, name, tokenized_folder)

    flat_golds = copy.deepcopy(golds)
    flat_predictions = copy.deepcopy(predictions)
    for i in range(len(flat_mark)):
        for j in range(len(flat_mark[i])):
            if flat_mark[i][j] == 0 and nested_mark[i][j] == 1:
                flat_golds[i][j] = "O"
                flat_predictions[i][j] = "O"

    nested_golds = copy.deepcopy(golds)
    nested_predictions = copy.deepcopy(predictions)
    for i in range(len(nested_mark)):
        for j in range(len(nested_mark[i])):
            if nested_mark[i][j] == 0 and flat_mark[i][j] == 1:
                nested_golds[i][j] = "O"
                nested_predictions[i][j] = "O"

    print("Flat events:")
    evaluate(flat_golds, flat_predictions)
    print()
    print("Nested events:")
    evaluate(nested_golds, nested_predictions)


def generate_flat_nested_mark(folder_name, name, tokenized_folder):
    input_path = folder_name + '/' + name + "/"
    file_names = get_file_names(input_path)

    flat_mark = []
    nested_mark = []
    cnt = 0
    for file_name in file_names:
        try:
            with open(folder_name + '/' + tokenized_folder + '/' + file_name + '.json', 'r') as json_file:
                json_obj = json.load(json_file)
                start_token_dict = json_obj["start_token_dict"]
                end_token_dict = json_obj["end_token_dict"]
                tokens = json_obj["tokens"]
                offsets = json_obj["offsets"]

            pre = 0
            start_sentence_offsets = []
            arr = []
            for i in range(len(tokens)):
                if tokens[i] == '.' or i == len(tokens) - 1:
                    start_sentence_offsets.append((offsets[pre][0], pre))
                    arr.append([0 for _ in range(i - pre + 1)])
                    pre = i + 1

            doc_flat_mark = copy.deepcopy(arr)     # Initialize
            doc_nested_mark = copy.deepcopy(arr)      # Initialize
            with open(input_path + file_name + '.a2', 'r') as file:
                trigger_dict = dict()
                while line := file.readline().rstrip():
                    if line[0] == "T":
                        parts = line.split("\t")
                        trigger_id = parts[0]
                        indicators = parts[1].split()
                        start = int(indicators[1])
                        end = int(indicators[2])
                        if str(start) not in start_token_dict:
                            print("NOT FOUND START:", start, file_name)
                        if str(end) not in end_token_dict:
                            print("NOT FOUND END:", end, file_name)

                        sent_index = upper_bound(start_sentence_offsets, start) - 1
                        start_index = start_token_dict[str(start)] - start_sentence_offsets[sent_index][1]
                        end_index = end_token_dict[str(end)] - start_sentence_offsets[sent_index][1]

                        trigger_dict[trigger_id] = []
                        for token_index in range(start_index, end_index + 1):
                            trigger_dict[trigger_id].append({
                                "sent_id": sent_index,
                                "token_index": token_index
                            })
                    elif line[0] == "E":
                        parts = line.split("\t")[1].split()
                        trigger_id = parts[0].split(":")[1]
                        isFlat = True
                        for i in range(1, len(parts)):
                            arg_id = parts[i].split(":")[1]
                            if arg_id[0] == "E":
                                isFlat = False
                                for trig in trigger_dict[trigger_id]:
                                    sent_index = trig["sent_id"]
                                    token_index = trig["token_index"]
                                    doc_nested_mark[sent_index][token_index] = 1
                        if isFlat:
                            for trig in trigger_dict[trigger_id]:
                                sent_index = trig["sent_id"]
                                token_index = trig["token_index"]
                                doc_flat_mark[sent_index][token_index] = 1

            flat_mark.extend(doc_flat_mark)
            nested_mark.extend(doc_nested_mark)
            cnt += 1
        except Exception as e:
            print(e)
            print("ERROR OCCURS!!")
            break

    print("Processed", cnt, "files!")
    return flat_mark, nested_mark


if __name__ == '__main__':
    generate_gold_labels("ge11", "dev", "tokenized_dev")
    # generate_flat_nested_mark("ge11", "dev", "tokenized_dev")
    evaluate_flat_nested_triggers("ge11", "dev", "tokenized_dev")
