import os
import uuid

from transformers import AutoTokenizer


def randomID():
    return str(uuid.uuid4())


def get_file_names(input_path):
    file_set = set()
    omitted_files = ["LICENSE", "README"]
    for file in os.listdir(input_path):
        if file in omitted_files:
            continue
        file_set.add(file.split(".")[0])

    file_names = list(file_set)
    file_names.sort()
    print("No of files:", len(file_names))
    print(file_names)
    return file_names


def processOffsets(obj, sentence):
    processed_offsets = []
    tokens = obj["ids_to_tokens"]
    offsets = obj["offset_mapping"]
    i = 0
    n = len(tokens)
    # hyperproduction, overexpression, superinduction
    split_list = ["production", "expression", "induction", "activation"]
    while i < n:
        start = offsets[i][0]
        end = offsets[i][1]
        i += 1
        while i < n and len(tokens[i]) > 2 and tokens[i][:2] == "##":
            end = offsets[i][1]
            i += 1
        tok = sentence[start:end]
        flag = False
        for word in split_list:
            if tok[-len(word):] == word and len(tok) > len(word):
                mid = end - len(word)
                processed_offsets.append((start, mid))
                processed_offsets.append((mid, end))
                flag = True
                break
        if not flag:
            processed_offsets.append((start, end))
    return processed_offsets


def bert_tokenize(sentence):
    # sentence = "Don't you love 🤗 Transformers? We sure do, we can do it. Hahaha."
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoded_input = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
    encoded_input['ids_to_tokens'] = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
    processed_offsets = processOffsets(encoded_input, sentence)
    tokens = []
    start_token_dict = dict()
    end_token_dict = dict()
    for i in range(len(processed_offsets)):
        start, end = processed_offsets[i]
        start_token_dict[start] = i
        end_token_dict[end] = i
        tokens.append(sentence[start:end])

    return start_token_dict, end_token_dict, tokens, processed_offsets


def lower_bound(arr, index, x):
    n = len(arr)
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if x <= arr[mid][index]:
            hi = mid
        else:
            lo = mid + 1
    if lo < n and arr[lo][index] < x:
        lo += 1
    return lo


# get index of first element (element 0 of pair) > x
def upper_bound(arr, x):
    n = len(arr)
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if x >= arr[mid][0]:
            lo = mid + 1
        else:
            hi = mid
    if lo < n and arr[lo][0] <= x:
        lo += 1
    return lo
