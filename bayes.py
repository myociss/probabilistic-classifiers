import os, math, sys
from collections import Counter


def get_vocabulary(item_class):
    class_vocabulary = []
    for filename in os.listdir('train/' + item_class):
        file = open(os.path.join('train/' + item_class, filename), encoding='latin-1')
        class_vocabulary += [word for line in file for word in line.split() if word.isalpha()]
    return class_vocabulary


def test(item_class):
    classifications = []
    for filename in os.listdir('test/' + item_class):
        file = open(os.path.join('test/' + item_class, filename), encoding='latin-1')
        # W ← EXTRACTTOKENSFROMDOC(V, d)
        tokens = [word for line in file for word in line.split() if word in all_vocab]

        # do score[c] ← log prior[c]
        score_ham = math.log(ham_prior)
        score_spam = math.log(spam_prior)
        
        for token in tokens:
            # do score[c] += log condprob[t][c]
            score_ham += math.log(ham_bayes_dict[token])
            score_spam += math.log(spam_bayes_dict[token])

        if score_ham > score_spam:
            classifications.append('ham')
        else:
            classifications.append('spam')
    return classifications
        


# Nc ← COUNTDOCSINCLASS(D, c)
spam_count = len(os.listdir('train/spam'))
ham_count = len(os.listdir('train/ham'))

# N ← COUNTDOCS(D)
total_doc_count = spam_count + ham_count

# prior[c] ← Nc/N

spam_prior = spam_count / total_doc_count
ham_prior = ham_count / total_doc_count

# textc ← CONCATENATETEXTOFALLDOCSINCLASS(D, c)

spam_vocab = get_vocabulary("spam")
ham_vocab = get_vocabulary("ham")

exclude_stopwords = sys.argv[1].split('=')[1]

# if the user wants to exclude stopwords
if exclude_stopwords == 'true':
    stopwords = [line.rstrip() for line in open('stopwords.txt')]
    spam_vocab = [word for word in spam_vocab if word not in stopwords]
    ham_vocab = [word for word in ham_vocab if word not in stopwords]


# V ← EXTRACTVOCABULARY(D)

all_vocab = spam_vocab + ham_vocab

# Tct ← COUNTTOKENSOFTERM(textc, t)

spam_vocab_count_dict = Counter(spam_vocab)
ham_vocab_count_dict = Counter(ham_vocab)

# do condprob[t][c] ← Tct+1/∑t′(Tct′+1)
spam_bayes_dict = { k: (spam_vocab_count_dict[k] + 1) / (len(spam_vocab) + len(spam_vocab_count_dict)) for k in all_vocab}
ham_bayes_dict = { k: (ham_vocab_count_dict[k] + 1) / (len(ham_vocab) + len(ham_vocab_count_dict)) for k in all_vocab}

spam_classifications = test('spam')
ham_classifications = test('ham')

spam_accuracy = spam_classifications.count('spam') / len(spam_classifications)
ham_accuracy = ham_classifications.count('ham') / len(ham_classifications)

spam_test_files_len = len(os.listdir('test/spam'))
ham_test_files_len = len(os.listdir('test/ham'))


spam_test_ratio = spam_test_files_len / (spam_test_files_len + ham_test_files_len)
ham_test_ratio = ham_test_files_len / (spam_test_files_len + ham_test_files_len)

total_accuracy = spam_accuracy * spam_test_ratio + ham_accuracy * ham_test_ratio

print("spam percent classified accurately: " + str(spam_accuracy))
print("ham percent classified accurately: " + str(ham_accuracy))

print("total accuracy: " + str(total_accuracy) )
