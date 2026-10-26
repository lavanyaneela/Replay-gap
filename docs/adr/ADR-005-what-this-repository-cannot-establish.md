# ADR-005: What this repository cannot establish

Status: accepted
Date: 2026-03-05

## Context

This is the boundary ADR. Everything above it argues for a choice; this one states what the
choices give up, so a reader does not have to work it out and a reviewer does not have to guess
whether the author knew.

## The assumption everything rests on, and the experiment that attacks it

Every corrected number here assumes the logged propensities are exact. They are, because the
logging policy is Plackett-Luce by construction and its propensities are a closed form. Almost
no production log is like that. Propensities get recorded at the wrong granularity, logged
before a reranking step nobody accounted for, or reconstructed months later from a checkpoint
that has moved.

E5 corrupts them deliberately and measures what happens. Two findings, and the second is the one
that matters.

First, the correction degrades. At a corruption factor of 3.0 the corrected estimate for the
most divergent policy is as wrong as the naive replay it was supposed to replace, and past that
it is worse.

Second, and this was not the expected result: the diagnostic does not merely fail to warn, it
moves the wrong way. Effective sample size reads 12.3% of the log at that crossover, higher than
it read on the uncorrupted log, and that happens in all 3 of the crossovers in the sweep.
Tempering flattens the weights, and flat weights are exactly what Kish's formula rewards. So the
number this tool uses to decide whether to trust an estimate climbs while the estimate it is
vouching for decays.

**Effective sample size measures the spread of the weights, not whether they are the right
weights.** No diagnostic in this repository detects a propensity that is confidently wrong.
There is no fix inside this design: detecting it needs an external signal, such as a holdout
with randomised serving, which this repository does not have and cannot simulate honestly.

## What the synthetic world costs

ADR-001 covers the choice. The boundary it creates:

- The feature distribution is simpler than any real catalogue. No long tail, no seasonality, no
  items that change, no user whose interests move inside a session.
- Retrieval is uniform and unbiased, which is the only confounder in a real system that has been
  deliberately removed. In production the retrieval stage is itself a policy trained on logged
  data, so the candidate set is confounded before the ranker sees it. Every number here would be
  worse with a biased retriever, not better.
- The click model is a position weighted Bernoulli with a fixed examination curve. Real users
  abandon, scroll, and are influenced by what sits next to a result.

## Which claims transfer, and which one does not

**Transfers unchanged, because it follows from the selection rule rather than from the data:**
replaying the deployed ranker against its own log reports a positive lift over itself. The
sessions a replay keeps are selected on agreement with the logging policy, and agreement with
the logging policy is correlated with reward in any log any ranker produced. The magnitude
depends on the system; the sign does not.

**Transfers unchanged, because it is structural:** naive replay cannot see the serving
temperature. It reads the first slot of the sessions where that slot already matched, and
nothing about how greedily the policy is served reaches either of those. Two policies that rank
identically and are served differently get the same naive estimate on any log.

**Transfers in mechanism, not in magnitude:** the effective sample size collapse, the sign
disagreement rate, the minimum detectable lift. These depend on how far the candidate policies
sit from the logging policy, which is a property of a particular system. A reader should take
these as evidence the effect is large, not as numbers to quote.

**Does not transfer, and genuinely needs a real log:** whether the effective sample size floor
of 2% is the right floor. That threshold is a policy decision calibrated against this world's
weight distribution. On a real log with a heavier tail it would be wrong, and there is no way to
calibrate it here. It is a configuration value for exactly this reason, and the repository does
not claim it generalises.

## Declared rather than observed

Three inputs are declared by the configuration rather than measured from anything:

- The examination curve. In production this is estimated, usually by an intervention, and the
  estimate is uncertain. Here it is a constant that both the world and the referee use, so
  position bias is known perfectly, which no real team enjoys.
- The candidate set size and the slate length, which are fixed at ten and three because the
  referee has to enumerate 720 slates (ADR-002).
- The overlap thresholds that decide a refusal, discussed above.

## Machine dependent figures

No absolute duration is published as a claim anywhere in this repository. Durations are recorded
in the experiment JSON for the record and never quoted in prose, because a published millisecond
is a fact about whichever machine ran it. Where a comparison genuinely concerns speed, the
repository publishes a ratio or a count.

## What would close each gap

| Gap | What would close it |
| --- | --- |
| Propensities assumed exact | A log with a randomised serving holdout, so weights can be validated against a known uniform arm |
| Unbiased retrieval | A retrieval stage trained on the logged data, making the candidate set confounded too |
| Examination curve known | Estimating it from the log with an interleaving experiment, and propagating its uncertainty |
| Refusal threshold uncalibrated | Running the tool against a real log with a known online result, and tuning the floor against outcomes |
| Short slate | A referee that does not enumerate, which means accepting sampling error in the referee |
