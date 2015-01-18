import numpy as np
from banditsolver import BanditSolver

class EpsilonGreedySolver (BanditSolver):
    def __init__(self, arms, epsilon=0.8):
        self.arms = arms
        self.epsilon = 0.8
        self.successes = np.zeros(arms)
        self.trials = np.zeros(arms)

        return

    def predict(self, *args):
        prev = np.seterr(all='raise')
        if np.random.random() < self.epsilon:
            try:
                return np.argmax(self.successes / self.trials)
            except FloatingPointError:
                pass # not enough data to say which is best
            finally:
                np.seterr(**prev)
        return np.random.random_integers(0, self.arms - 1)

    def train(self, arm, success, category_values):
        self.successes[arm] += 1 if success else 0
        self.trials[arm] += 1

