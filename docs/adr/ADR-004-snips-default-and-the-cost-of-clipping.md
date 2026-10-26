# ADR-004: SNIPS as the default estimator, and what the weight cap costs

Status: accepted
Date: 2026-03-04

## Context

Four estimators ship here: naive replay, plain inverse propensity scoring, self normalised
inverse propensity scoring, and a doubly robust estimator. One of them has to be the one the
tool's verdict and exit code are based on.

## Decision

SNIPS, with a configurable weight cap, and the share of sessions the cap binds on published next
to every estimate it affects.

## Why not plain IPS

IPS is unbiased when the propensities are exact, which they are here, and that is a genuine
argument for it. It is rejected on two grounds.

It is unbounded. The estimate is a mean of weighted rewards with no constraint tying it to the
range of rewards that actually occurred, so it can and does return values no slate could have
produced. On the greedily served policy in the default comparison it reports a lift of -9.95%
where the true lift is +21.53%, which is not an imprecise answer but a sign error of thirty
points.

Its variance is governed by the largest few weights. E3 measures this: on the most divergent
policy its error across replicates is several times that of any other estimator, and its bias is
near zero, which is the worst combination for a decision tool because it looks correct on
average and is unusable on any single run.

SNIPS divides by the sum of the weights rather than the sample size. It trades a small bias for
a large variance reduction and, more usefully, its estimate is bounded by the range of observed
rewards. A value estimate above the maximum achievable reward is a number no reviewer will read
past, and IPS produces them.

## Why not doubly robust, which usually wins

E3 says DR has the lowest error for most policies here, including the one it was expected to
lose on. It is not the default anyway, for one reason: it needs an outcome model, and the only
data available to fit that outcome model is the same confounded log.

The failure mode is specific and E3 measures it. For a policy served greedily, the slates it
wants to serve are slates the deployed ranker almost never served, so the outcome model is
extrapolating. On that policy SNIPS is at 3.78% error against 4.44% for the doubly robust
estimator, and the ordering reverses exactly where a team would most want to trust the answer.

DR is computed and published in every table. A reader who sees SNIPS and DR agree has more
reason to believe both, and a reader who sees them diverge has learned that the outcome model is
extrapolating, which is useful information rather than a problem.

## The weight cap, and what it costs

The cap is a configuration value, never a constant in the code. It bounds the variance of the
estimate at the cost of a bias, and the bias runs in the direction that flatters a candidate
policy: the sessions whose weights are cut are precisely the ones where the candidate most
disagreed with the log, so capping pulls the estimate towards the logged behaviour.

Two consequences are published rather than hidden. The share of sessions the cap actually bound
on appears next to any affected estimate, because a cap binding on one session in a thousand is
a footnote and one binding on one in five is the estimator. And the uncapped weights are kept
alongside the capped ones, so the plain IPS column in every table is computed without the cap,
which is what makes the comparison between them meaningful.

## The honest weakness of this choice

SNIPS is not unbiased and the repository publishes where it is wrong rather than only where it
wins. In the default comparison it calls a policy that is actually -0.05% a ship at +0.09%, at
97% effective sample size. That is a correct estimator behaving correctly on a question too fine
for the log to resolve, which is E4's subject, and it is in the README's first screen rather
than in a footnote.

## Consequences

- The verdict and exit code come from SNIPS, and the bootstrap interval around it is what
  decides between exit 0 and exit 1.
- Every table carries all four estimators, so a reader can check the default against the others.
- The cap is tuneable and its effect is always visible.
