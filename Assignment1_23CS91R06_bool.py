import pickle
import os

import sys

dictionary_path = sys.argv[1]
query_file_path = sys.argv[2]

if os.path.exists(dictionary_path):
    file = open(dictionary_path, 'rb')
    data = pickle.load(file)
    file.close()
else:
    exit("Inverted index file's path does not exist.")

if os.path.exists(query_file_path):
    file = open(query_file_path, 'r')
    lines = file.read().split('\n')
    lines.remove("")
    queries = {}
    for line in lines:
        filtered_list = line.split("    ")[1].split(" ")
        filtered_list.remove('')
        queries[line.split("    ")[0]] = filtered_list
    file.close()
else:
    exit("Query file's path does not exist.")


def merge(list1, list2):
    len1 = len(list1)
    len2 = len(list2)

    merged_list = []

    iterator1 = 0
    iterator2 = 0

    while iterator1 < len1 and iterator2 < len2:
        if list1[iterator1] == list2[iterator2]:
            merged_list.append(list1[iterator1])
            iterator1 = iterator1 + 1
            iterator2 = iterator2 + 1
        elif list1[iterator1] < list2[iterator2]:
            iterator1 = iterator1 + 1
        else:
            iterator2 = iterator2 + 1

    return merged_list


result = {}
for query in queries:
    result[query] = []

    # Handle query optimization by sorting on the basis of count of documents the word is occurring in
    # and store in a list of tuples as shown below.

    optimization_list = []
    for word in queries[query]:
        if word in data:
            optimization_list.append((len(data[word]), word))
        else:
            optimization_list.append((0, word))
        optimization_list.sort()

    # Iterate over each query in an optimized manner.
    # Initializing each query's result in result dictionary itself.
    # As the algorithm proceeds further, this dictionary will also reach its correct retrieval

    for index, sorted_words in enumerate(optimization_list):
        # If the first word itself has 0 occurrences, we don't need to iterate.
        if index == 0 and sorted_words[0] == 0:
            break
        # Initialize the answer with first word's occurrence
        elif index == 0:
            result[query] = data[sorted_words[1]]
        # Apply merge process
        else:
            result[query] = merge(result[query], data[sorted_words[1]])

# As using append functionality, deleting the output file if it already exists
if os.path.exists("Assignment1_23CS91R06_results.txt"):
    os.remove("Assignment1_23CS91R06_results.txt")

for i in result:

    # Open the file to write output in
    file = open('Assignment1_23CS91R06_results.txt', 'a')

    # Write output to file
    file.writelines(i + " : ")
    for word in result[i]:
        file.writelines(str(word) + " ")
    file.writelines("\n")

file.close()
