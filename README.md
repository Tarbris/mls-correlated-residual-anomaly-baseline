# Context-dependent residual anomaly workspace

Edit detector.py only. Implement class Detector with fit(rows) and
decision_function(rows). Each row has seven entries: a context indicator
0 or 1 followed by six finite real-valued sensor features. fit receives
900 unlabeled training rows with about 5% contamination. decision_function
returns a finite numeric score for each of 360 evaluation rows in the same
order; higher means more anomalous. The method receives no train/test labels,
latent signs, context-specific relation parameters, evaluator files or
reference implementation.

Normal rows have three strongly correlated adjacent sensor pairs. The
direction of each pair's relation reverses between the two contexts, and
the signs and offsets vary across episodes. Anomalies violate one pair's
context-specific relation by a moderate residual shift, while staying near
ordinary marginal ranges. Pooling contexts can cancel the useful dependence.
The evaluator computes area under the ROC curve (AUROC), with pairwise ties
counted as half; higher is better. Python 3.12 standard library and CPU
suffice. No network or GPU is needed.
