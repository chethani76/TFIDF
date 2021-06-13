#import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

doc1 = open("1", "r", encoding='utf-8').read()
doc2 = open("2", "r", encoding='utf-8').read()
doc3 = open("3", "r",encoding='utf-8').read()
doc4 = open("4", "r",encoding='utf-8').read()
doc5 = open("5", "r",encoding='utf-8').read()
query = open("query", "r", encoding='utf-8').read()

# instantiate the vectorizer object
# Tfidfvectorizer compute the word counts, idf and tf-idf values all at once.
tfidf = TfidfVectorizer()

# convert the documents into a matrix
# fit_transform do couple of things:
# first, it creates a dictionary of 'known' words based on the input text given to it. 
# Then it calculates the tf-idf for each term found in an article.
#lts in a matrix, where the rows are the individual Document files and the columns are the terms. 
# Thus, every cell represents the tf-idf score of a term in a file
response = tfidf.fit_transform([doc1, doc2, doc3, doc4, doc5, query]).todense()
print(response.shape)

# retrieve the specific terms and their tf-idf score found from the raw document
feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[0, col])

plagerism_percentage_dic = {}
query_vector = response[5]

# looping through document vectors to compute cosine similarity
for i in range(0,5):
    document_vector = response[i]
    plagerism_percentage_dic[f"Document {i + 1}"] = cosine_similarity(query_vector,document_vector)[0][0]

# print the document name with plagiarism count
print("\nPlagerismn counts of each document")
for (key, value) in plagerism_percentage_dic.items():
    print(f"{key}: {round(value * 100, 2)} %")

print("Document With higherst similarity to the query file : ", max(plagerism_percentage_dic, key=plagerism_percentage_dic.get))
