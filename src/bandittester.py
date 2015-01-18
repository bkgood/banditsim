import numpy as np
from thompsonsolver import ThompsonSolver
from epsilongreedysolver import EpsilonGreedySolver
from lwprsolver import LWPRSolver

class BanditTester (object):
    def __init__(self, arm_success_rates, n):
        self.categories = {}
        self.solvers = []
        self.arm_success_rates = arm_success_rates
        self.n = n

    def with_category(self, name, options):
        if len(self.solvers) > 0:
            raise Exception("no new categories once we've added a solver")

        self.categories[name] = options

    def with_solver(self, solver_cls, *args, **kwargs):
        self.solvers.append(solver_cls(len(self.arm_success_rates), *args, **kwargs))

    def test(self):
        regret = {}
        for s in self.solvers:
            regret[str(s)] = 0

        best_arm = np.argmax(self.arm_success_rates)
        visitors = np.random.rand(self.n)

        for v in visitors:
            for s in self.solvers: # we can parallelize over solvers
                arm = s.predict()
                success = False
                if self.arm_success_rates[arm] > v:
                    success = True
                s.train(arm, success, None)

                if arm != best_arm:
                    regret[str(s)] += rates[best_arm] - rates[arm]
        print regret
        print "done"
        #print ts.successes / (ts.successes + ts.failures)

if __name__ == '__main__':
    rates = np.array([0.1, 0.3, 0.8])
    tester = BanditTester(rates, 300)
    #tester.with_solver(ThompsonSolver)
    #tester.with_solver(EpsilonGreedySolver, epsilon=0.8)
    tester.with_solver(LWPRSolver, required_n=200)
    print "running test"
    tester.test()
