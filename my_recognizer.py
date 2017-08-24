import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # initialize final lists
    probabilities = []
    guesses = []
    # loop through test_set
    for X, lengths in test_set.get_all_Xlengths().values():
        # initialize attributes for selected items
        probabilities_temp = {}
        best_guess = None
        best_rec_score = float("-inf")
        # loop through models for respective word
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
                probabilities_temp[word] = logL
                # update score if higher; if TRUE, update guess
                if logL > best_rec_score:
                    best_rec_score = logL
                    best_guess = word
            except:
                probabilities_temp[word] = float("-inf")
            # append best guesses and probabilities
        guesses.append(best_guess)
        probabilities.append(probabilities_temp)
    return probabilities, guesses
