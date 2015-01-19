class BanditSolver (object):
    def __init__(self, arms):
        raise NotImplementedError()

    def with_categories(self, categories):
        raise NotImplementedError()

    def predict(self, category_values):
        raise NotImplementedError()

    def train(self, arm, success, category_values):
        raise NotImplementedError()

