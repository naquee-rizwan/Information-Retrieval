from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import sys
import os

query_set_path = sys.argv[1]

# Reading the query file.
# Before opening the file, checking if the file exists or not.
# If it does not exist, exiting with an error message on console.
if os.path.exists(query_set_path):
    with open(query_set_path, 'r') as file:
        # Splitting with an invalid string just to create a list of lines
        lines = file.read().split("\n##########\n")
    file.close()
else:
    exit("Query set file's path does not exist.")

for line in lines:
    query_metadata = line.split(".I ")
    query_metadata.remove('')

    # As using append functionality, deleting the output file if it already exists
    if os.path.exists("queries_23CS91R06.txt"):
        os.remove("queries_23CS91R06.txt")

    # Open the file to write output in
    file = open('queries_23CS91R06.txt', 'a')

    for index, string in enumerate(query_metadata):

        # Parsing the input data set
        split_query = string.split("\n.W\n")

        # Extract index from query
        index_of_query = split_query[0]

        # Extract words from query
        tokenized_words = word_tokenize(split_query[1])

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokenized_words = [word for word in tokenized_words if not word in stop_words]

        # Remove punctuations
        tokenized_words = [word for word in tokenized_words if not word in punctuation]

        # Applying WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = set()

        for tokens in tokenized_words:
            lemmatized_words.add(lemmatizer.lemmatize(tokens))

        # Write output to file
        file.writelines(index_of_query + "    ")
        for word in lemmatized_words:
            file.writelines(word + " ")
        file.writelines("\n")

    file.close()
