"""Weak marginal-distance starter; only this file is editable."""


class Detector:
    def fit(self, rows):
        features = [row[1:] for row in rows]
        n = len(features)
        self.mean = [sum(row[j] for row in features) / n for j in range(6)]
        self.scale = [(sum((row[j] - self.mean[j]) ** 2 for row in features) / n + 1e-8) ** 0.5
                      for j in range(6)]
        return self

    def decision_function(self, rows):
        return [sum(((row[j + 1] - self.mean[j]) / self.scale[j]) ** 2
                    for j in range(6)) for row in rows]
