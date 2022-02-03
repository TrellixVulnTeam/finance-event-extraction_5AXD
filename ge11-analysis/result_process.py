import json

import jsonlines

from utils import *


def processResults(folder_name, name, offset_file):
    input_path = folder_name + '/' + name + '/'
    file_names = get_file_names(input_path)

    filename_to_id = dict()
    fileid_to_tokenlist = dict()
    with jsonlines.open(folder_name + '/' + name + '.jsonl') as reader:
        for obj in reader:
            tokenid_to_pos = dict()
            for candidate in obj["candidates"]:
                tokenid_to_pos[candidate["id"]] = {
                    "sent_id": candidate["sent_id"],
                    "index": candidate["offset"][0]
                }
            fileid_to_tokenlist[obj["id"]] = tokenid_to_pos
            filename_to_id[obj["title"]] = obj["id"]
            # break

    with jsonlines.open(folder_name + '/results.jsonl') as reader:
        for obj in reader:
            fileid = obj["id"]
            if fileid in fileid_to_tokenlist:
                for token in obj["predictions"]:
                    tokenid = token["id"]
                    fileid_to_tokenlist[fileid][tokenid]["type_id"] = token["type_id"]

    with open(folder_name + '/' + offset_file) as json_file:
        offset_mark = json.load(json_file)

    cnt = 0
    for file_name in file_names:
        if file_name not in offset_mark:
            continue
        cnt += 1
        with open(input_path + file_name + '.txt', 'r') as file:
            text = file.read()
        fileid = filename_to_id[file_name]
        token_list = fileid_to_tokenlist[fileid]
        for key in token_list:
            sent_id = token_list[key]["sent_id"]
            index = token_list[key]["index"]
            type_id = token_list[key]["type_id"]
            offset_mark[file_name][sent_id][index]["type_id"] = type_id

        generateOutput(folder_name + "/output", text, file_name, offset_mark[file_name])

    print("Processed", cnt, "files!")


def generateOutput(output_folder, text, file_name, annotated_tokens):
    labels = ["None", "Gene_expression", "Positive_regulation", "Regulation", "Localization", "Phosphorylation",
              "Negative_regulation", "Transcription", "Entity", "Binding", "Protein_catabolism"]
    # annotated_tokens = [
    #     [
    #         {"token": "Resistance", "offset": [0, 10], "type_id": 6},
    #         {"token": "to", "offset": [11, 13], "type_id": 0},
    #     ]
    # ]

    lines = []
    cur_id = 1000
    for sentence in annotated_tokens:
        i = 0
        while i < len(sentence):
            type_id = sentence[i]["type_id"]
            if type_id == 0:
                i += 1
                continue
            pre = i
            while i + 1 < len(sentence) and type_id == sentence[i + 1]["type_id"]:
                i += 1
            start = sentence[pre]["offset"][0]
            end = sentence[i]["offset"][1]
            line = "T" + str(cur_id) + "\t" + labels[type_id] + " " + str(start) + " " + str(end) + "\t" + text[start:end]
            lines.append(line)
            i += 1
            cur_id += 1

    with open(output_folder + "/" + file_name + ".a2", 'w') as f:
        f.write("\n".join(lines))


def generateOffsetMapping(folder_name, name):
    input_path = folder_name + '/' + name + '/'
    file_names = get_file_names(input_path)

    json_obj = {}
    for file_name in file_names:
        with open(input_path + file_name + '.txt', 'r') as file:
            sentence = file.read()
            start_token_dict, end_token_dict, tokens, offsets = bert_tokenize(sentence)

        pre = 0
        offset_mark = []
        for i in range(len(tokens)):
            if tokens[i] == '.' or i == len(tokens) - 1:
                arr = []
                for index in range(pre, i + 1):
                    arr.append({
                        "token": tokens[index],
                        "offset": offsets[index]
                    })
                offset_mark.append(arr)
                pre = i + 1
        json_obj[file_name] = offset_mark

    with open(folder_name + '/' + name + '_offsets.json', "w") as outfile:
        json.dump(json_obj, outfile)

    print("Processed", len(json_obj), "files!")


def addEvent(root_folder, input_folder, output_folder):
    input_path = root_folder + '/' + input_folder + '/'
    output_path = root_folder + '/' + output_folder + '/'
    file_names = get_file_names(input_path)

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        os.makedirs(output_path)
        print("The output directory is created!")

    cnt = 0
    for file_name in file_names:
        cur_id = 1
        with open(input_path + file_name + '.a2', 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        event_lines = []
        for line in lines:
            parts = line.split("\t")
            trigger_id = parts[0]
            trigger_type = parts[1].split(" ")[0]
            if trigger_type == "Entity":
                continue
            event_text = "E" + str(cur_id) + "\t" + trigger_type + ":" + trigger_id + " Theme:T1"
            event_lines.append(event_text)
            cur_id += 1

        lines = lines + event_lines
        cnt += 1
        with open(root_folder + "/" + output_folder + "/" + file_name + ".a2", 'w') as f:
            f.write("\n".join(lines))

    print("Processed", cnt, "files!")


if __name__ == '__main__':
    generateOffsetMapping("ge11", "test")
    processResults("ge11", "test", "test_offsets.json")
    # generateOutput("ge11/output", "", "ABC", [])
    addEvent("ge11", "output", "output2")
