import numpy as np

def f1_score_from_cm(confusion):

    # Sum columns and rows
    sum_col = np.sum(confusion, axis=0)
    sum_row = np.sum(confusion, axis=1)
    diag = np.diag(confusion)

    # Macro precision/recall
    precision = diag / (sum_col.astype(float) + 1e-15)
    recall = diag / (sum_row.astype(float) + 1e-15)
    
    # Compute Macro F1
    f1_score_per_class = 2.*(precision * recall) / ((precision + recall) + 1e-15)
    macro_f1 = np.mean(f1_score_per_class)
    
    return macro_f1

def cohen_kappa_from_cm(confusion, weights=None):
    r"""Cohen's kappa: a statistic that measures inter-annotator agreement.
    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as
    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)
    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.
    Read more in the :ref:`User Guide <cohen_kappa>`.
    Parameters
    ----------
    confusion : confusion matrix
    weights : str, optional
        List of weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.
    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.

    """
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / float(np.sum(w_mat * expected))
    return 1 - k
    
def balanced_accuracy_score_from_cm(C, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / (C.sum(axis=1)).astype(float)
    if np.any(np.isnan(per_class)):
        #print('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1. / n_classes
        score -= chance
        score /= 1. - chance
    return score
