import numpy as np
import argparse
from sklearn.preprocessing import normalize

from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from WikiDataReader import *

#####################################################################
# Part 2:
# Define your new subclasses below!
#####################################################################

class CountDocumentVectors(WikiDocumentVectors):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)

    def train(self, documents):
        return self.vectorizer.fit_transform(documents)

    def get_document_vector(self, new_doc):
        return self.vectorizer.transform([new_doc])


class SVDDocumentVectors(WikiDocumentVectors):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)
        self.V = None

    def train(self, documents):
        tdm = self.vectorizer.fit_transform(documents)   # (N, 10000), sparse
        U, s, Vh = svds(tdm, k=100)                      # U (N, 100), s (100,), Vh (100, 10000)
        self.V = Vh.T                                    # (10000, 100)
        return U

    def get_document_vector(self, new_doc):
        query_tfidf = self.vectorizer.transform([new_doc])   # (1, 10000), sparse
        return query_tfidf @ self.V                          # (1, 100), dense




#####################################################################
# WikiSearchEngine
#####################################################################

class WikiSearchEngine():

    def __init__(self, articles, doc_vector_model):
        """constructor: stores 3 things, the articles, the model of the document vector, and the matrix of document vectors"""
        self.articles = articles
        self.doc_vector_model = doc_vector_model
        self.wiki_corpus_vectors = None

    def train_vector_model(self):
        """creates a generator and loops through it to store in the wiki_corpus_vectors matrix"""
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_corpus_vectors = self.doc_vector_model.train(self.articles.get_abstracts()) # loop over the generator, open and read the files, return as a matrix w 1 row per article
        self.wiki_corpus_vectors = normalize(self.wiki_corpus_vectors)
        print("Done!")

    def compute_rankings(self, query):
        """safety check to see if training happened"""
        if self.wiki_corpus_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        # model turns the query string into a ector, normalizes to length 1
        query_vector = normalize(self.doc_vector_model.get_document_vector(query))

        # REPLACE the following line with code that computes the
        # cosine similarity between the query vector and all document
        # vectors in the self.wiki_corpus_vectors matrix, and saves
        # the similarity scores to a list or numpy array called sims,
        # indexed in the same order as self.wiki_corpus_vectors.
        
        sims = self.wiki_corpus_vectors @ query_vector.T 
        if issparse(sims):
            sims = sims.toarray()
        sims = np.asarray(sims).ravel()
        #sims = []

        return sims, np.argsort(sims)[::-1]
        # artsort returns the indices that would sort the array from largest to smallest
        # [::-1] steps through the array backwards -> reverses the array

    def run_search(self, query, n_results=10):
        """calls compute_rankings and prints top 10 results"""
        print(f"Top {n_results} search results for query {query}:")
        sims, ranking = self.compute_rankings(query)
        for rank, i in enumerate(ranking[:n_results]): # keep the first 10 entries
            print(f"{rank+1}. {self.articles.get_names()[i]} (similarity={sims[i]})")
            # indexes positions by 1
            # get article title, and print




def main():
    """creates a parser to read the folder and other arguments"""
    parser = argparse.ArgumentParser() # turn 
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-q", "--query", type=str, metavar="Q", required=True,
                        help="The search query to run")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    parser.add_argument("-m", "--model", type=str, choices=["tfidf", "svd"], default="tfidf",
                        help="Which document vector model to use: tfidf or svd")
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    if args.model == "svd":
        doc_vector_model = SVDDocumentVectors()
    else:
        doc_vector_model = CountDocumentVectors()


    # doc_vector_model = None

    my_google = WikiSearchEngine(articles, doc_vector_model)
    my_google.train_vector_model()
    my_google.run_search(args.query)

if __name__ == "__main__":
    main()