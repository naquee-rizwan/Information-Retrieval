from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pickle
from string import punctuation

import sys

# Read inverted index's and CRAN folder's path passed from command line arguments
cran_folder_path = sys.argv[1]  # "cran"
inverted_index_path = sys.argv[2]  # "model_queries_23CS91R06.bin"

# Get cran folder's document and query file's paths
cran_document_path = os.path.join(cran_folder_path, "cran.all.1400")
cran_query_path = os.path.join(cran_folder_path, "cran.qry")

# Initialize corpus size. Vocabulary size will be updated later
CORPUS_SIZE = 1400

# Calculate document frequency for each term in the vocabulary
document_frequency = {}


def compute_document_frequency_from_inverted_index():
    # Read inverted index from pickle
    if os.path.exists(inverted_index_path):
        file = open(inverted_index_path, 'rb')
        inverted_index = pickle.load(file)
        file.close()
    else:
        exit("Inverted index pickle file does not exist. Please provide its path carefully from terminal.")

    for key in inverted_index:
        document_frequency[key] = len(inverted_index[key])

    assert (len(document_frequency) == len(inverted_index))


compute_document_frequency_from_inverted_index()
VOCABULARY_SIZE = len(document_frequency)

# Populate bag of words matrix for term frequency calculation
vectorized_documents = {}


def compute_vectorized_documents():
    # Reading the dataset file.
    # Before opening the file, checking if the file exists or not.
    # If it does not exist, exiting with an error message on console.
    if os.path.exists(cran_document_path):
        with open(cran_document_path, 'r') as file:
            # Splitting with an invalid string just to create a list of lines
            lines = file.read().split("\n##########\n")
        file.close()
    else:
        exit("Cran file with 1400 documents does not exist. Please provide Cran folder's path carefully from terminal.")

    for line in lines:
        document_metadata = line.split(".I ")
        document_metadata.remove('')

        for index, string in enumerate(document_metadata):

            # Extract index from document
            index_of_document = int(string.split("\n.T\n")[0])
            assert (index + 1 == index_of_document)

            # Extract words from document
            tokenized_words = word_tokenize(string.split("\n.W\n")[1])

            # Remove stop words
            stop_words = set(stopwords.words("english"))
            tokenized_words = [word for word in tokenized_words if not word in stop_words]

            # Remove punctuations
            tokenized_words = [word for word in tokenized_words if not word in punctuation]

            # Applying WordNet Lemmatizer
            lemmatizer = WordNetLemmatizer()

            # Make a count map for each document
            term_frequency_for_document = {}

            for token in tokenized_words:
                lemmatized_token = lemmatizer.lemmatize(token)
                assert (lemmatized_token in document_frequency)
                if lemmatized_token in term_frequency_for_document:
                    term_frequency_for_document[lemmatized_token] = term_frequency_for_document[lemmatized_token] + 1
                else:
                    term_frequency_for_document[lemmatized_token] = 1

            vectorized_documents[index_of_document] = term_frequency_for_document


compute_vectorized_documents()

# Populate bag of words matrix for term frequency calculation for queries
vectorized_queries = {}


def compute_vectorized_queries():
    # Reading the query file.
    # Before opening the file, checking if the file exists or not.
    # If it does not exist, exiting with an error message on console.
    if os.path.exists(cran_query_path):
        with open(cran_query_path, 'r') as file:
            # Splitting with an invalid string just to create a list of lines
            lines = file.read().split("\n##########\n")
        file.close()
    else:
        exit("Query file's path does not exist. Please provide Cran folder's path carefully from terminal.")

    for line in lines:
        query_metadata = line.split(".I ")
        query_metadata.remove('')

        for index, string in enumerate(query_metadata):

            # Parsing the input data set
            split_query = string.split("\n.W\n")

            # Extract index from query
            index_of_query = split_query[0]

            # Make a count map for each document
            term_frequency_for_query = {}

            # Extract words from query
            tokenized_words = word_tokenize(split_query[1])

            # Remove stop words
            stop_words = set(stopwords.words("english"))
            tokenized_words = [word for word in tokenized_words if not word in stop_words]

            # Remove punctuations
            tokenized_words = [word for word in tokenized_words if not word in punctuation]

            # Applying WordNet Lemmatizer
            lemmatizer = WordNetLemmatizer()

            for tokens in tokenized_words:
                lemmatized_token = lemmatizer.lemmatize(tokens)
                # Add the lemmatized only if it is present in inverted index
                if lemmatized_token in document_frequency:
                    if lemmatized_token in term_frequency_for_query:
                        term_frequency_for_query[lemmatized_token] = term_frequency_for_query[lemmatized_token] + 1
                    else:
                        term_frequency_for_query[lemmatized_token] = 1

            vectorized_queries[index_of_query] = term_frequency_for_query


compute_vectorized_queries()


def calculate_ranking_by_lnc_ltc_scheme():
    vectorized_documents_scheme_A = vectorized_documents.copy()
    vectorized_queries_scheme_A = vectorized_queries.copy()
    document_frequency_scheme_A = document_frequency.copy()

    # Computations for document vectors
    for document in vectorized_documents_scheme_A:
        vectorized_documents_scheme_A[document] = vectorized_documents_scheme_A[document].copy()
        cosine_normalization_document = 0.0
        for token in vectorized_documents_scheme_A[document]:
            vectorized_documents_scheme_A[document][token] = 1 + np.log10(
                vectorized_documents_scheme_A[document][token])
            cosine_normalization_document += vectorized_documents_scheme_A[document][token] ** 2
        for token in vectorized_documents_scheme_A[document]:
            vectorized_documents_scheme_A[document][token] /= np.sqrt(cosine_normalization_document)

    # Computations for query vectors
    for token in document_frequency_scheme_A:
        document_frequency_scheme_A[token] = np.log10(CORPUS_SIZE / document_frequency_scheme_A[token])

    for query in vectorized_queries_scheme_A:
        vectorized_queries_scheme_A[query] = vectorized_queries_scheme_A[query].copy()
        cosine_normalization_query = 0.0
        for token in vectorized_queries_scheme_A[query]:
            vectorized_queries_scheme_A[query][token] = 1 + np.log10(vectorized_queries_scheme_A[query][token])
            vectorized_queries_scheme_A[query][token] *= document_frequency_scheme_A[token]
            cosine_normalization_query += vectorized_queries_scheme_A[query][token] ** 2
        for token in vectorized_queries_scheme_A[query]:
            vectorized_queries_scheme_A[query][token] /= np.sqrt(cosine_normalization_query)

    # Calculate ranking by dot product of each query and each document and write output in the file
    file = open('Assignment2_23CS91R06_ranked_list_A.txt', 'w')

    for query in vectorized_queries_scheme_A:
        document_vector = []
        for document in vectorized_documents_scheme_A:
            dot_product = 0.0
            for token in vectorized_queries_scheme_A[query]:
                if token in vectorized_documents_scheme_A[document]:
                    dot_product += vectorized_queries_scheme_A[query][token] * vectorized_documents_scheme_A[document][
                        token]
            document_vector.append((-dot_product, document))
        document_vector.sort()

        # Write output to file
        file.writelines(query + " : ")
        for document in document_vector[:50]:
            file.writelines(str(document[1]) + " ")
        file.writelines("\n")

    file.close()


calculate_ranking_by_lnc_ltc_scheme()


def calculate_ranking_by_lnc_Ltc_scheme():
    vectorized_documents_scheme_B = vectorized_documents.copy()
    vectorized_queries_scheme_B = vectorized_queries.copy()
    document_frequency_scheme_B = document_frequency.copy()

    # Computations for document vectors
    for document in vectorized_documents_scheme_B:
        vectorized_documents_scheme_B[document] = vectorized_documents_scheme_B[document].copy()
        cosine_normalization_document = 0.0
        for token in vectorized_documents_scheme_B[document]:
            vectorized_documents_scheme_B[document][token] = 1 + np.log10(
                vectorized_documents_scheme_B[document][token])
            cosine_normalization_document += vectorized_documents_scheme_B[document][token] ** 2
        for token in vectorized_documents_scheme_B[document]:
            vectorized_documents_scheme_B[document][token] /= np.sqrt(cosine_normalization_document)

    # Computations for query vectors
    for token in document_frequency_scheme_B:
        document_frequency_scheme_B[token] = np.log10(CORPUS_SIZE / document_frequency_scheme_B[token])

    for query in vectorized_queries_scheme_B:
        vectorized_queries_scheme_B[query] = vectorized_queries_scheme_B[query].copy()
        cosine_normalization_query = 0.0

        average_term_frequency = 0.0
        for token in vectorized_queries_scheme_B[query]:
            average_term_frequency += vectorized_queries_scheme_B[query][token]
        average_term_frequency /= len(vectorized_queries_scheme_B[query])

        for token in vectorized_queries_scheme_B[query]:
            vectorized_queries_scheme_B[query][token] = (1 + np.log10(vectorized_queries_scheme_B[query][token])) / (
                        1 + np.log10(average_term_frequency))
            vectorized_queries_scheme_B[query][token] *= document_frequency_scheme_B[token]
            cosine_normalization_query += vectorized_queries_scheme_B[query][token] ** 2

        for token in vectorized_queries_scheme_B[query]:
            vectorized_queries_scheme_B[query][token] /= np.sqrt(cosine_normalization_query)

    # Calculate ranking by dot product of each query and each document and write output in the file
    file = open('Assignment2_23CS91R06_ranked_list_B.txt', 'w')

    for query in vectorized_queries_scheme_B:
        document_vector = []
        for document in vectorized_documents_scheme_B:
            dot_product = 0.0
            for token in vectorized_queries_scheme_B[query]:
                if token in vectorized_documents_scheme_B[document]:
                    dot_product += vectorized_queries_scheme_B[query][token] * vectorized_documents_scheme_B[document][
                        token]
            document_vector.append((-dot_product, document))
        document_vector.sort()

        # Write output to file
        file.writelines(query + " : ")
        for document in document_vector[:50]:
            file.writelines(str(document[1]) + " ")
        file.writelines("\n")

    file.close()


calculate_ranking_by_lnc_Ltc_scheme()


def calculate_ranking_by_anc_apc_scheme():
    vectorized_documents_scheme_C = vectorized_documents.copy()
    vectorized_queries_scheme_C = vectorized_queries.copy()
    document_frequency_scheme_C = document_frequency.copy()

    # Computations for document vectors
    for document in vectorized_documents_scheme_C:
        vectorized_documents_scheme_C[document] = vectorized_documents_scheme_C[document].copy()
        cosine_normalization_document = 0.0

        maximum_document_term_frequency = max(vectorized_documents_scheme_C[document].values()) if len(
            vectorized_documents_scheme_C[document].values()) > 0 else 0.0

        for token in vectorized_documents_scheme_C[document]:
            vectorized_documents_scheme_C[document][token] = 0.5 + (
                        0.5 * vectorized_documents_scheme_C[document][token]) / maximum_document_term_frequency
            cosine_normalization_document += vectorized_documents_scheme_C[document][token] ** 2

        for token in vectorized_documents_scheme_C[document]:
            vectorized_documents_scheme_C[document][token] /= np.sqrt(cosine_normalization_document)

    # Computations for query vectors
    for token in document_frequency_scheme_C:
        document_frequency_scheme_C[token] = max(0.0, np.log10((CORPUS_SIZE / document_frequency_scheme_C[token]) - 1))

    for query in vectorized_queries_scheme_C:
        vectorized_queries_scheme_C[query] = vectorized_queries_scheme_C[query].copy()
        cosine_normalization_query = 0.0

        maximum_query_term_frequency = max(vectorized_queries_scheme_C[query].values())

        for token in vectorized_queries_scheme_C[query]:
            vectorized_queries_scheme_C[query][token] = 0.5 + (
                        0.5 * vectorized_queries_scheme_C[query][token]) / maximum_query_term_frequency
            vectorized_queries_scheme_C[query][token] *= document_frequency_scheme_C[token]
            cosine_normalization_query += vectorized_queries_scheme_C[query][token] ** 2

        for token in vectorized_queries_scheme_C[query]:
            vectorized_queries_scheme_C[query][token] /= np.sqrt(cosine_normalization_query)

    # Calculate ranking by dot product of each query and each document and write output in the file
    file = open('Assignment2_23CS91R06_ranked_list_C.txt', 'w')

    for query in vectorized_queries_scheme_C:
        document_vector = []
        for document in vectorized_documents_scheme_C:
            dot_product = 0.0
            for token in vectorized_queries_scheme_C[query]:
                if token in vectorized_documents_scheme_C[document]:
                    dot_product += vectorized_queries_scheme_C[query][token] * vectorized_documents_scheme_C[document][
                        token]
            document_vector.append((-dot_product, document))
        document_vector.sort()

        # Write output to file
        file.writelines(query + " : ")
        for document in document_vector[:50]:
            file.writelines(str(document[1]) + " ")
        file.writelines("\n")

    file.close()


calculate_ranking_by_anc_apc_scheme()
