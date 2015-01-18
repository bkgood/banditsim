import lwpr
import numpy as np
from banditsolver import BanditSolver

# use LWPR ("Locally Weighted Projection Regression") to solve a
# bandit/contextual bandit problem.  To my knowledge, this is the first
# application of this model to the [contextual] bandit problem, and seems
# desirable for a number of reasons, listed on the webpages of a library
# implementing the model [1] Downsides have yet to surface, although by using a
# single model for all arms ("output dimensions") we "[assume] that all
# gradients of all outputs point roughly in the same direction." This doesn't
# seem good, although I haven't pondered it sufficiently to really claim one
# way or another.
# [1] http://wcms.inf.ed.ac.uk/ipab/slmc/research/software-lwpr

# Future plans include modeling the contextual bandit solution (done in private
# but not yet in this framework) and hopefully implementing another model for
# comparison. This current solution doesn't beat raw Thompson sampling, but of
# course does when conditions vary in such a way that Thompson sampling loses
# focus of the best predictions, a scecario easy to imagine in real-world,
# long-running bandits.

# Note that this requires the lwpr python module and this requires a patch (and
# a fair amount of patience) to build under MSVC, I can provide the patch on
# request and intend to send it upstream.

class LWPRSolver (BanditSolver):
    def __init__(self, arms, required_n):
        self.lwpr = lwpr.LWPR(1, arms)
        # erm, shouldn't this be a constant (as long as we are without
        # context)?
        self.x = np.zeros(arms) # plus category data
        self.y = np.zeros(arms)
        self.arms = arms
        # should try this per-input
        self.required_n = required_n

    def predict(self, *args):
        x = np.eye(1)
        (y, conf) = self.lwpr.predict_conf(x)
        #print y, conf
        #print y + np.abs(y - conf)
        if self.required_n > 0:
            return np.random.random_integers(0, self.arms - 1)
        #print y, conf
        return np.argmax(y + np.abs(y - conf))
        #return np.argmax(conf)
        #return np.random.random_integers(0, self.arms - 1)
        #return np.argmax(y)
        #return np.argmax(y + (1.0 - conf))

    def train(self, arm, success, category_values):
        x = np.eye(1)
        y = np.zeros(self.arms)
        y[arm] = 1 if success else 0
        self.lwpr.update(x, y)
        self.required_n -= 1
        return
