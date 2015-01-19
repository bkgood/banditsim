import numpy as np
from banditsim.bandittester import BanditTester
from banditsim.thompsonsolver import ThompsonSolver
from banditsim.epsilongreedysolver import EpsilonGreedySolver
from banditsim.lwprsolver import LWPRSolver

if __name__ == '__main__':
    rates = np.array([0.1, 0.3, 0.8])
    tester = BanditTester(rates, 30000)
    solvers = []
    for x in (
                tester.with_solver(ThompsonSolver),
                tester.with_solver(EpsilonGreedySolver, epsilon=0.8),
                tester.with_solver(LWPRSolver, required_n=50),
            ):
        solvers.append(x)

    print "running test"
    regret = tester.test()
    print "done"

    for solver in solvers:
        print "%s: %f" % (solver, regret[str(solver)])
