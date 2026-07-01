import numpy as np
from sklearn.metrics import classification_report
from ptgcl.cl import evaluate_cluster_labels as ptgcl_cl_evaluate_cluster_labels

def reorder_class_scores_ensure_micro_avg(scores, labels, ninstances):
    class_scores = { label: scores[label] for label in labels if label in scores }
    sum_support = sum(map(lambda cs: cs['support'], class_scores.values()))

    for label in labels:
        if label in scores:
            del scores[label]
    scores['class labels'] = {
        'n classes': len(labels),
        'n instances': ninstances,
        'class scores': class_scores,
        'support': sum_support
    }
    if not 'micro avg' in scores:
        if not 'accuracy' in scores:
            raise ValueError('Expected either Accuracy or Micro Avg to be in the scores, but found none')
        scores['micro avg'] = {'precision': scores['accuracy'], 'recall': scores['accuracy'], 'f1-score': scores['accuracy'], 'support': scores['macro avg']['support']}
    return scores

def novelty_detection(predicted, gold, known_labels, legacy_edits=True):
    if legacy_edits:
        gold_novel = [label in known_labels for label in gold]
        predicted_novel = [label.startswith('F') for label in predicted]
    else:
        gold_novel = gold
        predicted_novel = predicted
    assert len(gold_novel) == len(gold_novel)
    scores = classification_report(gold_novel, predicted_novel, labels=[True, False], target_names=['known', 'novel'], sample_weight=None, output_dict=True, zero_division=0)
    return reorder_class_scores_ensure_micro_avg(scores, ['known', 'novel'], len(gold_novel)) # {'acc': accuracy_score(gold_novel, predicted_novel), 'balancedcc': balanced_accuracy_score(gold_novel, predicted_novel)}

def known_frame_identification(predicted, gold, known_labels):
    # select only instances for which the gold label is known
    predinstances_with_known_label = [p_label for p_label, g_label in zip(predicted, gold) if g_label in known_labels]
    goldinstances_with_known_label = [g_label for g_label in gold if g_label in known_labels]
    predicted_known_labels = list(set(goldinstances_with_known_label))
    assert len(predinstances_with_known_label) == len(goldinstances_with_known_label)
    scores = classification_report(goldinstances_with_known_label, predinstances_with_known_label, labels=predicted_known_labels, target_names=None, sample_weight=None, output_dict=True, zero_division=0)
    return reorder_class_scores_ensure_micro_avg(scores, predicted_known_labels, len(goldinstances_with_known_label)) # accuracy_score(gold_known, predicted_known)

def frame_identification(predicted, gold, labels=None):
    return reorder_class_scores_ensure_micro_avg(classification_report(gold, predicted, labels=labels, target_names=None, sample_weight=None, output_dict=True, zero_division=0), labels, len(gold)) # accuracy_score(gold, predicted)

def unknown_frame_induction(predicted, gold, known_labels):
    # select only instances for which the gold label is unknown
    predinstances_with_unknown_label = [p_label for p_label, g_label in zip(predicted, gold) if g_label not in known_labels]
    goldinstances_with_unknown_label = [g_label for g_label in gold if g_label not in known_labels]

    if len(predinstances_with_unknown_label) < 1 and len(goldinstances_with_unknown_label) < 1:
        return { }
    
    predinstances_with_unknown_label = np.array(predinstances_with_unknown_label, dtype=object)
    goldinstances_with_unknown_label = np.array(goldinstances_with_unknown_label, dtype=object)

    cluster_eval_report = ptgcl_cl_evaluate_cluster_labels(
        labels_true=goldinstances_with_unknown_label,
        labels_pred=predinstances_with_unknown_label,
        omit_contingency=True)

    return cluster_eval_report

def frame_induction(predicted, gold):

    if len(predicted) < 1 and len(gold) < 1:
        return { }
    
    predicted_ = np.array(predicted, dtype=object)
    gold_ = np.array(gold, dtype=object)

    cluster_eval_report = ptgcl_cl_evaluate_cluster_labels(
        labels_true=gold_,
        labels_pred=predicted_,
        omit_contingency=True)

    return cluster_eval_report
    
def novelty_and_frame_identification(predicted, gold, known_labels, legacy_edits=True):
    if legacy_edits:
        gold_with_outlier = [frame_label if frame_label in known_labels else "outlier" for frame_label in gold]
    else:
        gold_with_outlier = gold
    if type(known_labels) == tuple:
        known_labels_ = known_labels + ('outlier',)
    elif type(known_labels) == list:
        known_labels_ = known_labels + ['outlier']
    else:
        raise ValueError(f'expected list of tuple as known_labels, not {type(known_labels).__name__}')
    #
    assert len(gold_with_outlier) == len(predicted)
    scores = classification_report(gold_with_outlier, predicted, labels=known_labels_, target_names=None, sample_weight=None, output_dict=True, zero_division=0)
    return reorder_class_scores_ensure_micro_avg(scores, known_labels_, len(gold_with_outlier)) # accuracy_score(gold_with_outlier, predicted)
