"""Open-horizon stop rule (2026-09-14, docs/prior_v2_plan.md 5.3): stop when the best value metric has not improved by
more than `delta` for `patience` consecutive validations. delta 0 = off, so fixed-step runs are unchanged. State is not
carried across a resume (the first validation after a resume starts a fresh window)."""


class Plateau:
    def __init__(self, delta, patience):
        self.delta, self.patience, self.best, self.since = delta, patience, float("inf"), 0

    def step(self, v):   # -> True when the run should stop
        if v < self.best - self.delta:
            self.best, self.since = v, 0
        else:
            self.since += 1
        return self.delta > 0 and self.since >= self.patience
