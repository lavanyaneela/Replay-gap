# ADR-002: Compute ground truth by enumerating slates, not by Monte Carlo

Status: accepted
Date: 2026-03-02

## Context

Given a world and a candidate policy, the referee has to produce that policy's expected reward
over the context and candidate distribution the log contains. The usual way is Monte Carlo: draw
slates from the policy, draw clicks from the world, average, and quote a standard error.

## Decision

Enumerate every ordered slate and integrate the click model analytically. Choose the action
space so this is affordable: ten candidates per session, three slate positions, which is 720
ordered slates.

## Why

A sampled referee would put sampling error on both sides of every comparison this repository
makes. The headline claim is that an estimator is wrong by a measurable amount, and if the
quantity it is wrong about is itself an estimate, every result carries two error bars and the
interesting one is the smaller.

With a Plackett-Luce policy the slate distribution is a closed form, and with a Bernoulli click
model the expectation of the reward is its probability. So the policy's value is a finite sum
over 720 terms, computed exactly. The referee contributes no sampling error at all. "Naive
replay got the sign wrong" becomes a statement about a known number rather than about a
confidence interval overlapping zero.

The cost is a constraint on the design: the slate has to stay short enough to enumerate. Ten
candidates and three positions is 720; ten and five would be 30,240, and a hundred candidates
and ten positions is past any machine. That constraint is accepted deliberately, and it is why
the action space is small rather than because small was easier.

## Alternatives considered

**Monte Carlo with a large sample.** Simple, and it scales to any slate length. Rejected because
the sampling error does not vanish, it only gets small, and the comparisons here are between
estimators whose differences are sometimes a fraction of a percent. A referee that is itself
uncertain cannot adjudicate those.

**Analytic expectation with an independence assumption.** Treating positions as independent
makes the sum trivial at any slate length. Rejected because it is wrong: a slate is a sampling
without replacement, so an item in the first slot cannot also be in the second, and ignoring
that biases the referee in the same direction for every policy, which is the hardest kind of
error to notice.

**Enumerate only the high probability slates.** Truncating the sum would allow longer slates.
Rejected because the truncation error depends on the policy's temperature, so a sharply served
policy would be refereed more accurately than a stochastic one, and the comparison between them
is exactly what E3 is about.

## Implementation notes

Permutation probabilities are computed in a (permutations, sessions) layout and transposed back.
Gathering 720 whole rows out of a (candidates, sessions) array is a run of contiguous copies,
while the same gather in the natural layout is strided. On a fifty thousand session log the two
layouts measured four to one, and the referee runs on every policy in every experiment, so the
difference is the difference between a suite that reruns in CI and one that does not.

Collapsing the slate distribution onto positions is a dense matrix product against a fixed
indicator per position rather than a scatter add, for the same reason and with a similar factor.

## Consequences

- Every "true lift" figure in this repository is exact to floating point, and the tests assert
  it against an independent Monte Carlo rather than assuming it.
- The slate length is a fixed constraint of the design and is documented as one.
- A reader extending this to a realistic slate length has to replace the referee, and the README
  says so under future work.
