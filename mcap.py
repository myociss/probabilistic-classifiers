import os, random, math, sys


def get_vocabulary(item_class):
    class_vocabulary = []
    for filename in os.listdir('train/' + item_class):
        file = open(os.path.join('train/' + item_class, filename), encoding='latin-1')
        class_vocabulary += [word for line in file for word in line.split() if word.isalpha()]
    return class_vocabulary

def test(item_class):
    classifications = []
    for filename in os.listdir('test/' + item_class):
        data = Data_Vector(filename, all_vocab, item_class, 'test')
        data.set_probability_spam(weights)
        if 1 - data.probability_spam == 0 or data.probability_spam / (1 - data.probability_spam) > 1:
            classifications.append('ham')
        else:
            classifications.append('spam')
    return classifications


class Data_Vector:
    def __init__(self, filename, vocabulary, classification, data_type):
        self.filename = filename
        self.classification = classification
        self.data_type = data_type
        vocab = self.extract_vocab()
        self.features = { word: 1 if word in vocab else 0 for word in vocabulary  }
        self.probability_spam = 0
        if self.classification == 'ham':
            self.Y = 0
        else:
            self.Y = 1

    def extract_vocab(self):
        file = open(os.path.join(self.data_type + '/' + self.classification, self.filename), encoding='latin-1')
        return [word for line in file for word in line.split()]

    def set_probability_spam(self, w):
        try:
            exp = math.exp(sum([w.get(feature) * value for feature, value in self.features.items()  ]))
        except OverflowError:
            exp = float('inf')
        self.probability_spam = 1 / (1 + exp)

lam = float(sys.argv[1].split('=')[1])
learning_rate = float(sys.argv[2].split('=')[1])
iterations = int(sys.argv[3].split('=')[1])
exclude_stopwords = sys.argv[4].split('=')[1]


data = []

spam_vocab = get_vocabulary("spam")
ham_vocab = get_vocabulary("ham")

if exclude_stopwords == 'true':
    stopwords = [line.rstrip() for line in open('stopwords.txt')]
    spam_vocab = [word for word in spam_vocab if word not in stopwords]
    ham_vocab = [word for word in ham_vocab if word not in stopwords]

all_vocab = set(spam_vocab + ham_vocab)


for filename in os.listdir('train/spam'):
    data.append(Data_Vector(filename, all_vocab, 'spam', 'train'))

for filename in os.listdir('train/ham'):
    data.append(Data_Vector(filename, all_vocab, 'ham', 'train'))



weights = { word: 0 for word in all_vocab }


for x in range(iterations):
    gradient = {}
    for e in data:
        e.set_probability_spam(weights)


    for word in all_vocab:
        cost_term = sum([ example.features.get(word) * (example.Y - (1 - example.probability_spam)) for example in data])
        gradient[word] = cost_term


    for word in all_vocab:
        gradient[word] = gradient.get(word) - lam * weights.get(word) ** 2
        weights[word] = weights.get(word) + learning_rate * gradient.get(word)
       

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
