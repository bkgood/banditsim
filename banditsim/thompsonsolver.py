import numpy as np
from banditsolver import BanditSolver

# References:
# http://www.jmlr.org/proceedings/papers/v23/agrawal12/agrawal12.pdf
# http://www.economics.uci.edu/~ivan/asmb.874.pdf

class ThompsonSolver (BanditSolver):
    def __init__(self, arms):
        self.arms = arms
        self.successes = np.zeros(arms)
        self.failures = np.zeros(arms)

        self.predictor = np.vectorize(
            lambda a,b: np.random.beta(a+1, b+1)
        )

        return

    def predict(self, *args):
        return np.argmax(
            self.predictor(self.successes, self.failures)
        )

    def train(self, arm, success, category_values):
        if success:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

    def __str__(self):
        return "Thompson Solver"
