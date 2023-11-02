import os

import sys

# Read Gold standard text file and retrieved document's text file for each scheme
gold_standard_text_file_argument = "cran/cranqrel"  # sys.argv[1]
retrieved_documents_text_file_argument = "Assignment2_23CS91R06_ranked_list_A.txt"  # sys.argv[2]

# Reading the gold standard text file and our retrieved responses' text file.

# Before opening the file, checking if the file exists or not.
# If it does not exist, exiting with an error message on console.
if os.path.exists(retrieved_documents_text_file_argument):
    file = open(retrieved_documents_text_file_argument, 'r')
    lines = file.read().split('\n')
    lines.remove("")
    retrieved_documents_from_queries = {}
    for index, line in enumerate(lines):
        filtered_list = line.split(":")[1].split(" ")
        filtered_list.remove('')
        filtered_list.remove('')
        assert (len(filtered_list) == 50)
        for index_of_item, item in enumerate(filtered_list):
            filtered_list[index_of_item] = int(item)
            assert (filtered_list[index_of_item] == int(item))

        retrieved_documents_from_queries[index + 1] = {"query_id": line.split(":")[0], "retrieved_documents": filtered_list}
    file.close()
else:
    exit("Retrieved document's file (Assignment2_23CS91R06_ranked_list_<K>.txt) does not exist. Please "
         "provide path carefully from terminal.")

assert (len(retrieved_documents_from_queries) == 225)

# Before opening the file, checking if the file exists or not.
# If it does not exist, exiting with an error message on console.
if os.path.exists(gold_standard_text_file_argument):
    file = open(gold_standard_text_file_argument, 'r')
    lines = file.read().split('\n')
    gold_standard_documents = {}
    for line in lines:
        splitted_line = line.split(" ")
        while splitted_line.__contains__(''):
            splitted_line.remove('')
        assert (len(splitted_line) == 3)
        if int(splitted_line[0]) not in gold_standard_documents:
            gold_standard_documents[int(splitted_line[0])] = []
        if splitted_line[2] != '-1':
            gold_standard_documents[int(splitted_line[0])].append((-int(splitted_line[2]), int(splitted_line[1])))

    file.close()
else:
    exit("Cran query response file with Gold standard results does not exist. Please provide path carefully from "
         "terminal.")

assert (len(retrieved_documents_from_queries) == 225)

for query in gold_standard_documents:
    gold_standard_documents[query].sort()
    for indexed_item, items in enumerate(gold_standard_documents[query]):
        items_copy = [items[1], -items[0]]
        gold_standard_documents[query][indexed_item] = (items_copy[0], items_copy[1])

print()
