import numpy as np
import argparse
from sklearn.preprocessing import normalize

from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from WikiDataReader import *

#####################################################################
# Part 3:
# Define your new subclasses below!
#####################################################################

class SVDWordVectors(WikiWordVectors):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)
        self.V = None

    def train(self, documents):
        tdm = self.vectorizer.fit_transform(documents)   # (N, 10000), sparse
        U, s, Vh = svds(tdm, k=100)                      # U (N, 100), s (100,), Vh (100, 10000)
        self.V = Vh.T                                    # (10000, 100): row j = word j
        return self.V

    def get_word_index(self, word):
        return self.vectorizer.vocabulary_.get(word, -1)

    def get_vocabulary(self):
        return list(self.vectorizer.get_feature_names_out())

    def get_word_vector(self, word):
        i = self.get_word_index(word)
        if i == -1:
            return None
        return self.V[i:i+1]                             # (1, 100)




#####################################################################
# WikiWordLookup
#####################################################################

class WikiWordLookup():

    def __init__(self, articles, word_vector_model):
        self.articles = articles
        self.word_vector_model = word_vector_model
        self.wiki_word_vectors = None

    def train_vector_model(self):
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_word_vectors = self.word_vector_model.train(self.articles.get_abstracts())
        self.wiki_word_vectors = normalize(self.wiki_word_vectors)
        print("Done!")

    def compute_rankings(self, input_word):
        if self.wiki_word_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        word_vector = self.word_vector_model.get_word_vector(input_word)
        if word_vector is None:
            return None, None
        word_vector = normalize(word_vector)

        # TODO: REPLACE the following line with code that computes the
        # cosine similarity between the vector for input_word and all
        # other word vectors in the self.wiki_word_vectors matrix, and
        # saves the similarity scores to a list or numpy array called 
        # sims, indexed in the same order as self.wiki_word_vectors.
        sims = self.wiki_word_vectors @ word_vector.T
        if issparse(sims):
            sims = sims.toarray()
        sims = np.asarray(sims).ravel()

        return sims, np.argsort(sims)[::-1][1:]

    def run_search(self, input_word, n_results=10):
        input_word = input_word.lower()
        print(f"Top {n_results} most similar words to {input_word}:")
        sims, ranking = self.compute_rankings(input_word)
        if ranking is None:
            print(f"'{input_word}' not found in the vocabulary!")
        else:
            for rank, i in enumerate(ranking[:n_results]):
                print(f"{rank+1}. {self.word_vector_model.get_vocabulary()[i]} (similarity={sims[i]})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-w", "--word", type=str, metavar="W", required=True,
                        help="The word to look up")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    # TODO replace the following line with an initialization of the appropriate WikiWordVectors subclass
    word_vector_model = SVDWordVectors()

    word_lookup = WikiWordLookup(articles, word_vector_model)
    word_lookup.train_vector_model()
    word_lookup.run_search(args.word)

if __name__ == "__main__":
    main()