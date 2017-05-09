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

    return

    for i in labels:
        for j in labels:
            if i == j:
                true_positives[i] += cm[i,j]
            else:
                false_negatives[i] += cm[i,j]
                false_positives[j] += cm[i,j]

    sb = ''
    for value, count in true_positives.most_common():
        s = '{0}: {1}, '.format(value, count)
        sb += s
    log.info("True Positives: %d (%s)", sum(true_positives.values()), sb)
    
    for value, count in false_negatives.most_common():
        s = '{0}: {1}, '.format(value, count)
        sb += s
    log.info("False Negatives: %d (%s)", sum(false_negatives.values()), sb)

    for value, count in false_positives.most_common():
        s = '{0}: {1}, '.format(value, count)
        sb += s
    log.info("False Positives: %d (%s)", sum(false_positives.values()), sb)


    log.info("F Scores:")
    for i in sorted(labels):
        if true_positives[i] == 0:
            fscore = 0
        else:
            precision = true_positives[i] / float(true_positives[i]+false_positives[i])
            recall = true_positives[i] / float(true_positives[i]+false_negatives[i])
            fscore = 2 * (precision * recall) / float(precision + recall)

        log.info("%s %f", i, fscore)

if __name__ == "__main__":
    expected = 'DET NN VB DET JJ NN NN IN DET NN DET NN VB DET JJ NN NN IN DET NN'.split()
    predicted = 'DET VB VB DET NN NN NN IN DET NN DET NN NN DET NN NN NN IN DET NN'.split()
    labels = 'DET NN VB IN JJ'.split()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    report(expected, predicted, labels, logger)
