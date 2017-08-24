import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_bic_score = float("inf")
        best_model = self.base_model(self.n_constant)
        for _ in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(_)
                logL = model.score(self.X, self.lengths)
                logN = math.log(_)
                # formula for p, given _ states and features of the _th model
                p = _ ** 2 + 2 * _ * model.n_features - 1
                new_bic_score = -2 * logL + p * logN
                # is new_score lower than existing best option?
                if new_bic_score < best_bic_score:
                    best_bic_score = new_bic_score
                    best_model = model
            except:
                continue
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_dic_score = float("-inf")
        best_model = None
        scores = []
        try:
            for _ in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(_)
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        scores.append(model.score(X, lengths))
                    else:
                        pass
                    new_dic_score = model.score(self.X, self.lengths) - np.mean(scores)
                    if new_dic_score > best_dic_score:
                        best_dic_score = new_dic_score
                        best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = self.base_model(self.n_constant)
        folds = KFold(n_splits=4)
        for _ in range(self.min_n_components, self.max_n_components+1):
            try:
                scores = []
                model = self.base_model(_)
                for train_idx, test_idx in folds.split(self.sequences):
                    self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                    trained = self.base_model(_)
                    X, lengths = combine_sequences(test_idx, self.sequences)
                    scores.append(trained.score(X, lengths))
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
            except:
                continue
        return best_model
