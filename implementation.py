from collections import Counter
from math import log

import numpy as np

text = '''Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
all the king's horses and all the king's men
couldn't put Humpty together again'''

def main(text):
    # tasks your code should perform:

    # 1. split the text into words, and get a list of unique words that appear in it
    # a short one-liner to separate the text into sentences (with words lower-cased to make words equal
    # despite casing) can be done with
    # docs = [line.lower().split() for line in text.split('\n')]

    # 2. go over each unique word and calculate its term frequency, and its document frequency

    # 3. after you have your term frequencies and document frequencies, go over each line in the text and
    # calculate its TF-IDF representation, which will be a vector

    # 4. after you have calculated the TF-IDF representations for each line in the text, you need to
    # calculate the distances between each line to find which are the closest.

    documents = [line.lower().split() for line in text.split('\n')]
    num_of_documents = len(documents)
    print("Documents:")
    print(documents)
    print('---')

    # Obtain unique words.
    unique_words = []
    for document in documents:
        for word in document:
            if word not in unique_words:
                unique_words.append(word)
    print("Unique words:")
    print(unique_words)
    print('---')

    # Create dictionary to store all document frequencies.
    df = {}
    for word in unique_words:
        df[word] = None
    print("Dictionary to store all documents frequencies:")
    print(df)
    print('---')

    # Calculate document frequencies.
    for word in df.keys():
        num_of_appearances = 0
        for document in documents:
            if word in document:
                num_of_appearances += 1
        document_frequency = num_of_appearances / num_of_documents
        df[word] = document_frequency
    print("Document frequencies:")
    print(df)
    print('---')

    # Create a dictionary to store all term frequencies.
    tf = {}
    for document in range(num_of_documents):
        unique_words_in_current_document = []
        for word in documents[document]:
            if word not in unique_words_in_current_document:
                unique_words_in_current_document.append(word)
        term_frequencies_for_document = {}
        for word in unique_words_in_current_document:
            term_frequencies_for_document[word] = None
        tf[document] = term_frequencies_for_document
    print("Dictionary to store all term frequencies:")
    print(tf)
    print('---')

    # Calculate term frequencies.
    for document in range(num_of_documents):
        frequencies = Counter(documents[document])
        for term in tf[document].keys():
            tf[document][term] = frequencies[term] / len(documents[document])
    print("Term frequencies:")
    print(tf)
    print('---')

    # Create a dictionary to store all term frequency inverse document frequncy values.
    tf_idf = {}
    for document in range(num_of_documents):
        unique_words_in_current_document = []
        for word in documents[document]:
            if word not in unique_words_in_current_document:
                unique_words_in_current_document.append(word)
        term_frequencies_for_document = {}
        for word in unique_words_in_current_document:
            term_frequencies_for_document[word] = None
        tf_idf[document] = term_frequencies_for_document
    print("Dictionary to store all tf-idf values:")
    print(tf_idf)
    print('---')

    # Calculate tf-idf values.
    for document in range(num_of_documents):
        for term in tf_idf[document].keys():
            tf_idf[document][term] = tf[document][term] * log(1 / df[term], 10)
    print("All tf-idf values:")
    print(tf_idf)
    print('---')

    # Put tf-idf values into a list of lists.
    tf_idf_list = []
    for document in range(num_of_documents):
        current_list = [0 for word in unique_words]
        for term in tf_idf[document].keys():
            if term in unique_words:
                current_list[unique_words.index(term)] = tf_idf[document][term]
        tf_idf_list.append(current_list)
    print("tf-idf list:")
    print(tf_idf_list)
    print('---')

    # Find most similar docs.
    distances = np.empty((num_of_documents, num_of_documents), dtype=np.float)
    num_of_terms = len(tf_idf_list[0])
    for document in range(num_of_documents):
        for other_document in range(document, num_of_documents):
            if other_document == document:
                distances[document, other_document] = np.inf
            else:
                distance = sum([abs(tf_idf_list[other_document][term] - tf_idf_list[document][term]) for term in range(num_of_terms)])
                distances[document, other_document] = distance
                distances[other_document, document] = distance
    print(np.unravel_index(np.argmin(distances), distances.shape))



main(text)
