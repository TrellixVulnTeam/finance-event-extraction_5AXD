import json

import jsonlines
from utils import *


def test_preprocess(folder_name, input_name, output_name, tokenized_folder=None):
    input_path = folder_name + '/' + input_name + '/'
    file_names = get_file_names(input_path)

    # file_names = ["PMC-1920263-03-MATERIALS_AND_METHODS-02"]
    json_arr = []
    # Number of test files: 347
    for file_name in file_names:
        if tokenized_folder is None:
            with open(input_path + file_name + '.txt', 'r') as file:
                sentence = file.read()
                start_token_dict, end_token_dict, tokens, offsets = bert_tokenize(sentence)
        else:
            with open(input_path + file_name + '.txt', 'r') as file:
                sentence = file.read()
            with open(folder_name + '/' + tokenized_folder + '/' + file_name + '.json', 'r') as json_file:
                json_obj = json.load(json_file)
                tokens = json_obj["tokens"]
                offsets = json_obj["offsets"]

        json_obj = {}
        json_obj['id'] = randomID()
        json_obj['title'] = file_name
        json_obj['content'] = []
        pre = 0
        sent_cnt = 0
        for i in range(len(tokens)):
            if tokens[i] == '.' or i == len(tokens) - 1:
                json_obj['content'].append({
                    "sentence": sentence[offsets[pre][0]:offsets[i][1]],
                    "tokens": tokens[pre:(i + 1)]
                })
                pre = i + 1
                sent_cnt += 1

        json_obj['candidates'] = []
        for sent_index in range(len(json_obj['content'])):
            token_list = json_obj['content'][sent_index]["tokens"]
            for token_index in range(len(token_list)):
                obj = {
                    "id": randomID(),
                    "trigger_word": token_list[token_index],
                    "sent_id": sent_index,
                    "offset": [token_index, token_index + 1]
                }
                json_obj['candidates'].append(obj)

        json_arr.append(json_obj)

    # print(json.dumps(json_arr))
    with jsonlines.open(folder_name + '/' + output_name + '.jsonl', mode='w') as writer:
        writer.write_all(json_arr)

    print("Processed", len(json_arr), "files!")


def train_preprocess(folder_name, name):
    input_path = folder_name + '/' + name + "/"

    with open(folder_name + '/labels.json') as json_file:
        event_type_dict = json.load(json_file)
    cnt = len(event_type_dict)

    file_names = get_file_names(input_path)

    # file_names = ["PMC-1310901-01-INTRODUCTION"]
    json_arr = []
    with jsonlines.open(folder_name + '/' + name + '.jsonl') as reader:
        for obj in reader:
            json_arr.append(obj)

    # Number of train files: 908
    # Number of dev files: 259
    batch_index = 0
    batch_size = 227
    # file_names[(batch_index * batch_size):(batch_index * batch_size + batch_size)]
    for file_name in file_names:
        try:
            with open(input_path + file_name + '.txt', 'r') as file:
                sentence = file.read()
                start_token_dict, end_token_dict, tokens, offsets = bert_tokenize(sentence)

            json_obj = {}
            json_obj['id'] = randomID()
            json_obj['title'] = file_name
            json_obj['content'] = []
            pre = 0
            start_sentence_offsets = []
            for i in range(len(tokens)):
                if tokens[i] == '.' or i == len(tokens) - 1:
                    start_sentence_offsets.append((offsets[pre][0], pre))
                    json_obj['content'].append({
                        "sentence": sentence[offsets[pre][0]:offsets[i][1]],
                        "tokens": tokens[pre:(i+1)]
                    })
                    pre = i + 1

            json_obj['events'] = []
            mark = set()
            with open(input_path + file_name + '.a2', 'r') as file:
                while line := file.readline().rstrip():
                    if line[0] != "T":
                        continue
                    indicators = line.split("\t")[1].split()
                    event_type = indicators[0]
                    start = int(indicators[1])
                    end = int(indicators[2])
                    if start not in start_token_dict:
                        print("NOT FOUND START:", start, file_name)
                    if end not in end_token_dict:
                        print("NOT FOUND END:", end, file_name)

                    sent_index = upper_bound(start_sentence_offsets, start) - 1
                    start_index = start_token_dict[start] - start_sentence_offsets[sent_index][1]
                    end_index = end_token_dict[end] - start_sentence_offsets[sent_index][1]

                    if event_type not in event_type_dict:
                        event_type_dict[event_type] = cnt
                        cnt += 1
                    token_list = json_obj['content'][sent_index]["tokens"]
                    for token_index in range(start_index, end_index + 1):
                        mark.add((sent_index, token_index))
                        obj = {
                            "id": randomID(),
                            "type": event_type,
                            "type_id": event_type_dict[event_type],
                            "mention": [{
                                "id": randomID(),
                                "trigger_word": token_list[token_index],
                                "sent_id": sent_index,
                                "offset": [token_index, token_index + 1]
                            }]
                        }
                        json_obj['events'].append(obj)

            json_obj['negative_triggers'] = []
            for sent_index in range(len(json_obj['content'])):
                token_list = json_obj['content'][sent_index]["tokens"]
                for token_index in range(len(token_list)):
                    if (sent_index, token_index) in mark:
                        continue
                    obj = {
                        "id": randomID(),
                        "trigger_word": token_list[token_index],
                        "sent_id": sent_index,
                        "offset": [token_index, token_index + 1]
                    }
                    json_obj['negative_triggers'].append(obj)

            json_arr.append(json_obj)
        except Exception as e:
            print(e)
            print("ERROR OCCURS!!")
            break

    # print(json.dumps(json_arr))
    with jsonlines.open(folder_name + '/' + name + '.jsonl', mode='w') as writer:
        writer.write_all(json_arr)

    with open(folder_name + '/' + 'labels.json', 'w') as f:
        json.dump(event_type_dict, f)

    print("DONE BATCH", batch_index)
    print("Processed", len(json_arr), "files!")


def saveTokenize(root_folder, name, output_folder):
    input_path = root_folder + '/' + name + "/"
    output_path = root_folder + '/' + output_folder + '/'
    file_names = get_file_names(input_path)

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        os.makedirs(output_path)
        print("The output directory is created!")

    cnt = 0
    for file_name in file_names:
        try:
            with open(input_path + file_name + '.txt', 'r') as file:
                sentence = file.read()
                start_token_dict, end_token_dict, tokens, offsets = bert_tokenize(sentence)
            json_obj = {
                "tokens": tokens,
                "offsets": offsets,
                "start_token_dict": start_token_dict,
                "end_token_dict": end_token_dict
            }
            with open(output_path + '/' + file_name + '.json', 'w') as f:
                json.dump(json_obj, f)
            cnt += 1
        except Exception as e:
            print(e)
            print("ERROR OCCURS!!")
            break

    print("Processed", cnt, "files!")


if __name__ == '__main__':
    train_preprocess("ge11", "train")
    train_preprocess("ge11", "dev")
    test_preprocess("ge11", "test", "test")

    saveTokenize("ge11", "dev", "tokenized_dev")
    test_preprocess("ge11", "dev", "dev_test", "tokenized_dev")
