# ADR-001: A seeded synthetic world, not a public learning to rank dataset

Status: accepted
Date: 2026-03-02

## Context

This repository makes one kind of claim: that a given offline estimator is wrong by a specific
amount. Every such claim needs a referee, something that knows the true online value of a policy
the log never ran.

The obvious first choice is a public learning to rank dataset. MSLR-WEB30K, Yahoo LTR and
Istella are all well known, carry graded relevance labels, and would let a reader say "this is
real data" rather than "this is a simulation". That is a real advantage and it was the reason
this decision needed writing down rather than assuming.

## Decision

Use a seeded synthetic world with known true relevance, a known logging policy, and exact
propensities.

## Why the public datasets do not work for this question

A relevance label is not a counterfactual. Those datasets record how relevant a document was
judged to be, not what a user would have done if a different ranker had served them a different
slate, and the difference is the whole subject here.

Concretely, three things are missing and none of them can be recovered:

1. **No logging policy.** The datasets are label sets, not interaction logs. There is no
   deployed ranker whose decisions produced them, so there is nothing for a candidate policy to
   be confounded by, and the bias this repository measures does not exist in them.
2. **No propensities.** Without the probability the deployed policy assigned to the slate it
   served, importance weighting is not defined. They would have to be invented, at which point
   the dataset is synthetic in the one dimension that matters and real only in the features.
3. **No position bias.** The labels are per document, not per impression, so the examination
   curve has to be assumed anyway.

A common workaround is to simulate a logging policy over a public dataset and treat the graded
labels as the click model. That is a reasonable thing to do and it is still a simulation: the
part that would be real, the features, is the part this repository does not depend on, and the
part that would be synthetic, the policy and the propensities, is the part every number rests
on. It buys the appearance of realism and none of the substance.

## What this costs, stated plainly

The feature distribution here is simpler than a real catalogue. True relevance is a bilinear
function of a user vector and an item vector, users sit in segments, and the click model is a
position weighted Bernoulli. Real catalogues have long tails, seasonality, items that change,
and users whose interests move within a session. None of that is here.

The claims in this repository are about the *structure* of offline evaluation, not about the
magnitude of the effect in any particular product. A reader should take "naive replay overstates
by this much on this world" as a demonstration that the mechanism exists and is large, not as a
number to quote for their own system.

## What a reader should check instead

The mechanism, not the magnitude. The deployed ranker replaying against its own log reports a
positive lift over itself, and its true lift over itself is zero by construction, not by
simulation: the policy and the referee are the same object. That particular number would be the
same in any world, because it follows from the selection rule rather than from the data.

ADR-005 lists which claims here would transfer to a real log unchanged and which one genuinely
needs a real log to establish.

## Consequences

- Every figure is reproducible by anybody with the repository and one command, with no download
  and no network.
- The referee is exact rather than sampled (ADR-002), which is not possible with any real data.
- The repository must say, in the README and not in a footnote, that the world is synthetic and
  what that means for each claim. It does, under "What this cannot establish".
