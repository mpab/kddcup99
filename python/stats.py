from collections import Counter
from nltk.metrics import ConfusionMatrix
import logging

def report(expected, predicted, labels, log):
    
    cm = ConfusionMatrix(expected, predicted)

    log.info("Confusion matrix:\n%s", cm)

    log.info("Confusion matrix: sorted by count\n%s", cm.pretty_format(sort_by_count=True))
      
    true_positives = Counter()
    false_negatives = Counter()
    false_positives = Counter()
    missing_labels = Counter()

    #merge expected & predicted, & get unique values
    tested_labels = set(expected + predicted)

    for i in tested_labels:
        for j in tested_labels:
            if i == j:
                true_positives[i] += cm[i,j]
            else:
                false_negatives[i] += cm[i,j]
                false_positives[j] += cm[i,j]

    sb = ''
    for value, count in true_positives.most_common():
        s = '{0}={1}, '.format(value, count)
        sb += s
    log.info("True Positives (%d): %s\n", sum(true_positives.values()), sb)
    
    sb = ''
    for value, count in false_negatives.most_common():
        s = '{0}={1}, '.format(value, count)
        sb += s
    log.info("False Negatives (%d): %s\n", sum(false_negatives.values()), sb)

    sb = ''
    for value, count in false_positives.most_common():
        s = '{0}={1}, '.format(value, count)
        sb += s
    log.info("False Positives (%d): %s\n", sum(false_positives.values()), sb)

    sb = ''
    last = len(tested_labels) - 1
    for i, x in enumerate(sorted(tested_labels)):
        if true_positives[x] == 0:
            fscore = 0
        else:
            precision = true_positives[x] / float(true_positives[x]+false_positives[x])
            recall = true_positives[x] / float(true_positives[x]+false_negatives[x])
            fscore = 2 * (precision * recall) / float(precision + recall)

        if i != last:
            sb += '{0}={1}, '.format(x, fscore)
        else:
            sb += '{0}={1}'.format(x, fscore)

    log.info('F Scores: {0}\n'.format(sb))

    untested_labels = set(labels) - tested_labels

    if (len(untested_labels)):
        log.info('No F Scores for untested categories: {0}\n'.format(list(untested_labels)))

if __name__ == "__main__":
    expected = 'DET NN VB DET JJ NN NN IN DET NN DET NN VB DET JJ NN NN IN DET NN'.split()
    predicted = 'DET VB VB DET NN NN NN IN DET NN DET NN NN DET NN NN NN IN DET NN'.split()
    labels = 'DET NN VB IN JJ'.split()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    report(expected, predicted, labels, logger)
