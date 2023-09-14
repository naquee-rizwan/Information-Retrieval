from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import pickle
import os
import sys

dataset_path = sys.argv[1]

# Reading the dataset file.
# Before opening the file, checking if the file exists or not.
# If it does not exist, exiting with an error message on console.
if os.path.exists(dataset_path):
    with open(dataset_path, 'r') as file:
        # Splitting with an invalid string just to create a list of lines
        lines = file.read().split("\n##########\n")
    file.close()
else:
    exit("Dataset file's path does not exist.")

for line in lines:
    document_metadata = line.split(".I ")
    document_metadata.remove('')

    dictionary = {}
    for index, string in enumerate(document_metadata):

        # Extract index from document
        index_of_document = int(string.split("\n.T\n")[0])
        assert(index + 1 == index_of_document)

        # Extract words from document
        tokenized_words = word_tokenize(string.split("\n.W\n")[1])

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokenized_words = [word for word in tokenized_words if not word in stop_words]

        # Remove punctuations
        tokenized_words = [word for word in tokenized_words if not word in punctuation]

        # Applying WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = set()

        for token in tokenized_words:
            lemmatized_token = lemmatizer.lemmatize(token)
            if lemmatized_token in dictionary:
                dictionary[lemmatized_token].add(index_of_document)
            else:
                dictionary[lemmatized_token] = {index_of_document}

    for key in dictionary:
        dictionary[key] = sorted(set(dictionary[key]))

    if os.path.exists("model_queries_23CS91R06.bin"):
        os.remove("model_queries_23CS91R06.bin")

    file = open("model_queries_23CS91R06.bin", 'wb')
    pickle.dump(dictionary, file)
    file.close()
