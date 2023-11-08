import os

import sys

import numpy as np

# Read Gold standard text file and retrieved document's text file for each scheme
gold_standard_text_file_argument = sys.argv[1]
retrieved_documents_text_file_argument = sys.argv[2]

# Reading the gold standard text file and our retrieved responses' text file.

# Before opening the file, checking if the file exists or not.
# If it does not exist, exit with an error message on console.
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

        retrieved_documents_from_queries[index + 1] = {"query_id": line.split(":")[0],
                                                       "retrieved_documents": filtered_list}
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
        # Taking negative just for ease of decreasing order sorting
        items_copy = [items[1], -items[0]]
        gold_standard_documents[query][indexed_item] = (items_copy[0], items_copy[1])

assert (len(retrieved_documents_from_queries) == len(gold_standard_documents))

name_of_output_file = "Assignment2_23CS91R06_metrics_"
if retrieved_documents_text_file_argument.__contains__("A.txt"):
    name_of_output_file += "A.txt"
elif retrieved_documents_text_file_argument.__contains__("B.txt"):
    name_of_output_file += "B.txt"
elif retrieved_documents_text_file_argument.__contains__("C.txt"):
    name_of_output_file += "C.txt"

file = open(name_of_output_file, 'w')


def calculate_average_precision(_query_index, k):
    retrieved = retrieved_documents_from_queries[query_index]["retrieved_documents"][:k]
    gold_label = gold_standard_documents[query_index]

    number_of_documents = 0
    total_precision = 0.0

    for _index, docID in enumerate(retrieved):
        for gold_docID in gold_label:
            if docID == gold_docID[0]:
                number_of_documents += 1
                total_precision += number_of_documents / (_index + 1)

    return round(total_precision / number_of_documents, 4) if number_of_documents > 0 else 0.0


file.writelines("Query-wise Average Precision (AP) @10\n\n")

average_precision_over_queries_10 = 0.0
for query_index in range(1, len(retrieved_documents_from_queries) + 1):
    # Calculate ranking by dot product of each query and each document and write output in the file
    average_precision = calculate_average_precision(query_index, 10)
    average_precision_over_queries_10 += average_precision
    file.writelines(retrieved_documents_from_queries[query_index]["query_id"] + " : ")
    file.writelines(str(average_precision))
    file.writelines("\n")

file.writelines("\n--------------------\n\n")

file.writelines("Query-wise Average Precision (AP) @20\n\n")

average_precision_over_queries_20 = 0.0
for query_index in range(1, len(retrieved_documents_from_queries) + 1):
    # Calculate ranking by dot product of each query and each document and write output in the file
    average_precision = calculate_average_precision(query_index, 20)
    average_precision_over_queries_20 += average_precision
    file.writelines(retrieved_documents_from_queries[query_index]["query_id"] + " : ")
    file.writelines(str(average_precision))
    file.writelines("\n")

file.writelines("\n--------------------\n\n")


def calculate_ndcg(_query_index, k):
    retrieved = retrieved_documents_from_queries[query_index]["retrieved_documents"][:k]
    gold_label = gold_standard_documents[query_index]

    gold_label_relevance = []
    for __item in gold_label:
        gold_label_relevance.append(__item[1])

    while len(gold_label_relevance) < len(retrieved):
        gold_label_relevance.append(0)

    gold_label_relevance = gold_label_relevance[:k]

    retrieved_document_relevance = []

    while len(retrieved_document_relevance) < len(retrieved):
        retrieved_document_relevance.append(0)

    for _index, docID in enumerate(retrieved):
        for gold_docID in gold_label:
            if docID == gold_docID[0]:
                retrieved_document_relevance[_index] = gold_docID[1]
                break

    assert (len(retrieved_document_relevance) == len(gold_label_relevance))
    for __index, _ in enumerate(retrieved_document_relevance):
        if __index == 0:
            continue
        else:
            retrieved_document_relevance[__index] /= np.log2(__index + 1)
            gold_label_relevance[__index] /= np.log2(__index + 1)
            retrieved_document_relevance[__index] += retrieved_document_relevance[__index-1]
            gold_label_relevance[__index] += gold_label_relevance[__index-1]

    return round(retrieved_document_relevance[k-1] / gold_label_relevance[k-1], 4)


file.writelines("Query-wise Normalized Discounted Cumulative Gain (NDCG) @10\n\n")

average_ndcg_over_queries_10 = 0.0
for query_index in range(1, len(retrieved_documents_from_queries) + 1):
    # Calculate ranking by dot product of each query and each document and write output in the file
    ndcg = calculate_ndcg(query_index, 10)
    average_ndcg_over_queries_10 += ndcg
    file.writelines(retrieved_documents_from_queries[query_index]["query_id"] + " : ")
    file.writelines(str(ndcg))
    file.writelines("\n")

file.writelines("\n--------------------\n\n")

file.writelines("Query-wise Normalized Discounted Cumulative Gain (NDCG) @20\n\n")

average_ndcg_over_queries_20 = 0.0
for query_index in range(1, len(retrieved_documents_from_queries) + 1):
    # Calculate ranking by dot product of each query and each document and write output in the file
    ndcg = calculate_ndcg(query_index, 20)
    average_ndcg_over_queries_20 += ndcg
    file.writelines(retrieved_documents_from_queries[query_index]["query_id"] + " : ")
    file.writelines(str(ndcg))
    file.writelines("\n")

file.writelines("\n--------------------\n\n")

file.writelines("Mean Average Precision (AP) @10 :"
                + str(round(average_precision_over_queries_10 / len(retrieved_documents_from_queries), 4)) + "\n")
file.writelines("Mean Average Precision (AP) @20 : "
                + str(round(average_precision_over_queries_20 / len(retrieved_documents_from_queries), 4)) + "\n")
file.writelines("Mean Normalized Discounted Cumulative Gain (NDCG) @10 : "
                + str(round(average_ndcg_over_queries_10 / len(retrieved_documents_from_queries), 4)) + "\n")
file.writelines("Mean Normalized Discounted Cumulative Gain (NDCG) @20 : "
                + str(round(average_ndcg_over_queries_20 / len(retrieved_documents_from_queries), 4)) + "\n")

file.writelines("\n--------------------\n")

file.close()
