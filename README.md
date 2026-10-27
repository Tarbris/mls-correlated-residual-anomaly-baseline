# Context-dependent residual anomaly workspace

Edit detector.py only. Implement class Detector with fit(rows) and
decision_function(rows). Each row has seven entries: a context indicator
0 or 1 followed by six finite real-valued sensor features. fit receives
900 unlabeled training rows with about 12% contamination. decision_function
returns a finite numeric score for each of 360 evaluation rows in the same
order; higher means more anomalous. The method receives no train/test labels,
latent signs, context-specific relation parameters, evaluator files or
reference implementation.

Normal rows have three strongly correlated adjacent sensor pairs. The
direction of each pair's relation reverses between the two contexts. The
signs, offsets, and residual noise scales vary across episodes; three quite
different noise scales are assigned to different pairs in an unknown order.
Anomalies violate one pair's context-specific relation by an amount tied to
that pair's own noise scale. Raw residual magnitude therefore cannot rank
all pair anomalies fairly. Pooling contexts also cancels useful dependence.
The evaluator computes area under the ROC curve (AUROC), with pairwise ties
counted as half; higher is better. Python 3.12 standard library and CPU
suffice. No network or GPU is needed.
