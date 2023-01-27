import pandas as pd
import numpy as np

data=pd.read_csv(r'C:\Users\Fidha\Desktop\Sem3 datasets\india-news-headlines.csv')
data
data.head()
data.tail()
data.shape
data.info()
data.describe()
data.isnull()
data.isnull().sum()
data.duplicated()
data.duplicated().sum()
data.drop_duplicates(inplace=True)
data.duplicated().sum()
from sklearn.feature_extraction.text import TfidfVectorizer

text_content = data['headline_text']
vector = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents
                             #min_df=8,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
tfidf = vector.fit_transform(text_content)

data = pd.DataFrame(tfidf[0].T.todense(), index = vector.get_feature_names(), columns=["TF-IDF"])

data = df.sort_values('TF-IDF', ascending=False)

data

from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(query):
    query_vec = vector.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    return similarity

# Get user query
query = input("Enter your query: ")

# Ranking: Rank the news headlines based on their cosine similarity scores to the user query
s = get_similarity(query)
ranking = s.argsort()[:-5:-1]

# Return: Return the top 10 most relevant news headlines
for i in ranking:
    print(text_content[i])

