# Defense guide: replay-gap

Every number in this document is re-measured by `make receipts` and checked against the sentence
around it, so what is written here is what the code currently produces.

## The 30 second version

> Teams validate a new ranker by replaying it against the production log. I measured how wrong
> that is, against a referee that knows the true answer exactly. Replaying the deployed ranker
> against its own log says it is **+5.53% better than itself**, and the tool I built reports
> when a log cannot answer the question at all.

Then stop.

The question that usually comes next is "how do you know the true answer?" The answer is one
sentence: the world is simulated, so the slate distribution is enumerable and the click model is
analytic, which makes ground truth a closed form rather than an estimate. Then go to ADR-001 if
they push on the simulation.

## The claims, and how each is proved

### Claim 1: replaying the deployed ranker against its own log overstates it

```bash
replay-gap evaluate incumbent --no-ranker
```

Naive replay lift +5.53%, against a true lift of exactly +0.00%, because the policy being evaluated
and the policy that produced the log are the same object. The figure was computed on 17,187
sessions, 34.4% of the log.

**If pushed on "that is just one policy":** E1 samples 180 candidate rankers uniformly over the
simplex of true relevance, the incumbent's opinion, and an arbitrary wrong direction. Naive
replay picks the wrong winner in 27.1% of decisive cases, and came in above the truth in 96% of
scenarios.

**If pushed on "you designed the incumbent to fail":** that is the point of it. The incumbent is
not a designed failure, it is an identity: it is the logging policy, so its true lift over
itself is zero by construction rather than by measurement. Any non zero estimate is pure
estimator bias with no model quality mixed in. That is why it is the headline rather than a more
interesting policy.

### Claim 2: the failure is one sided, and that was not the prediction

```bash
python tools/run_experiments.py --only e1_sign_agreement
```

The plan for this repository predicted a symmetric failure: a replay flatters a policy that
agrees with production and punishes one that disagrees. The measurement disagreed. Naive replay
overstates in 96% of scenarios, with a mean signed error of +5.04% against SNIPS at +0.33%.

**The reason, which is the part worth saying:** a replay discards the sessions where the
candidate disagreed, rather than down weighting them. So there is no population of sessions left
to drag a divergent policy's estimate down. Selection, not weighting, is what makes it one
sided.

**If pushed:** the worst case in that sweep is a ranker whose true lift is -10.80% read as
+6.24%, a gap of 17.04%. A team acting on that ships a model that costs them.

### Claim 3: a replay cannot see the serving temperature at all

```bash
python tools/run_experiments.py --only e3_estimator_comparison
```

Two policies that rank identically, differing only in how greedily they are served. Their true
values differ by 16.39%. A replay returns the same number for both, to every digit.

**The mechanism:** a replay keeps the sessions where the candidate's top choice was the one
served, and averages their reward. Neither step involves anything below the first slot, and
serving temperature changes the distribution over slates rather than the top choice. So the
estimate is structurally blind to it.

**Why this is the strongest version of the argument:** it does not depend on the simulation at
all. It follows from what the estimator reads.

### Claim 4: correcting the estimate costs sample size, and sometimes all of it

```bash
replay-gap evaluate contrarian_sharp --no-ranker   # exits 2
```

Effective sample size 810 of 50,000 logged sessions, which is 1.62% of the log. The tool refuses rather than
estimating. Had it answered, SNIPS would have reported +25.97% where the true lift is +21.53%,
and plain IPS reports -9.95%, the wrong sign by thirty points.

**If pushed on "isn't refusing just giving up":** refusing is the feature. An evaluation tool
with no way to decline will always produce a number, and a number is what gets shipped on. Exit
code 2 exists so a caller can tell "the tool worked and the answer is no" from "this log cannot
answer that".

### Claim 5: the corrected estimator is not a fix either, and the repository says so

On the default policy set, SNIPS calls a policy whose true lift is -0.05% a ship at +0.09%, at
97% of the log still effective. The log cannot resolve a lift that small. E4 quantifies it: the
median minimum detectable lift across the policy set is a median of 2.09%, and for 5 of 6
policies it exceeds the one percent a team would typically ship on.

## Questions that are meant to be hard

### "Isn't this just a wrapper around standard off-policy estimators?"

The estimators are textbook and the README says so. Three things here are not.

The referee is exact. Every comparison in this repository is against a closed form value rather
than another estimate, which is what turns "IPS is theoretically biased" into "IPS got the sign
wrong by thirty points on this policy".

The tool refuses. Standard libraries return a number and sometimes a warning. This one has a
refusal path with its own exit code, because a number next to a warning gets read and the
warning does not.

The receipts are mechanical. Every figure in every document is re-measured by CI and checked
against the sentence around it, and `docs/receipts-proof.md` is the transcript of the checker
failing on five deliberate corruptions.

### "The world is synthetic. Does any of this transfer?"

Some of it, and the README says which. Take it claim by claim.

**Transfers unchanged:** the deployed ranker overstating itself. It follows from the selection
rule rather than from the data, so the sign holds on any log any ranker produced. The magnitude
does not.

**Transfers unchanged:** a replay's blindness to serving temperature. Structural, as above.

**Transfers in mechanism only:** the collapse rates, the disagreement rate, the detectable
lifts. These depend on how far candidates sit from production, which is a property of a
particular system.

**Does not transfer:** whether a 2% overlap floor is the right floor. It is calibrated against
this world's weight distribution, which is why it is a configuration value and not a constant,
and the repository does not claim it generalises.

I also considered using a public learning to rank dataset and rejected it. Those are label sets,
not interaction logs: no logging policy, no propensities, no position bias. The usual workaround
is to simulate a logging policy over them, which makes the part I depend on synthetic and the
part I do not depend on real. ADR-001 is that argument in full.

### "What is the weakest part?"

Two answers, and the second one is about me rather than about the tool.

The propensities are exact here, and in production they almost never are. That is the single
assumption everything rests on.

I attacked it rather than noting it. E5 corrupts the logged propensities deliberately. The
correction stops being worth having at a corruption factor of 3.0, which is a real limit. But
the result that matters is the second one, and it was worse than I expected: **the diagnostic
moves the wrong way.** Effective sample size reads 12.3% of the log at that crossover, higher
than it read on the uncorrupted log, and that happens in all 3 of the crossovers in the sweep.
Tempering flattens the weights and flat weights are what Kish's formula rewards, so the number I
use to decide whether to trust an estimate climbs while the estimate decays.

There is no fix inside this design. Detecting it needs an external signal, such as a holdout
with randomised serving, which this repository does not have and cannot simulate honestly. It is
the first item under future work for that reason.

The second weakest part is the verification discipline, and I would rather say it than have it
found. Three separate times a fix landed in the code and the edit describing it silently failed
to apply, and twice I reported something fixed without checking the artifact. The regression
test that passed against its own bug is the same error one layer down. What I take from it is
that the receipts mechanism is not decoration: this repository publishes a number nobody can
quietly change precisely because I am demonstrably capable of believing a change happened when
it did not.

The second weakest part is that retrieval is uniform and unbiased, which is the one confounder
in a real system I deliberately removed. In production the retrieval stage is itself trained on
logged data, so the candidate set is confounded before the ranker sees it. Every number here
would be worse with a biased retriever, not better.

### "What would you do differently with more time?"

Validate the propensities instead of assuming them, which means a randomised serving holdout
giving a known uniform arm to check the weights against. That is the extension E5 argues for and
it would change what the tool can claim, not just how well it claims it.

I would also confound the retrieval stage, and I would replace the referee. Enumerating 720
slates caps the slate length at three, which is fine for the argument and wrong for a realistic
product surface. A longer slate needs a sampled referee and an honest account of its error,
which is a real trade against the exactness this version is built on.

### "Why SNIPS as the default when your own experiment says DR usually wins?"

Because DR needs an outcome model and the only data to fit it on is the same confounded log. E3
measures where that bites: for a policy served greedily, SNIPS is at 3.78% error against 4.44%
for the doubly robust estimator, because DR's direct term is extrapolating onto slates
production almost never served. The ordering reverses exactly where a team would most want to
trust it.

DR is computed and published in every table anyway. Seeing SNIPS and DR agree means the outcome
model is in distribution; seeing them diverge means it is extrapolating. That is information,
and collapsing it into a single recommended number is the feature I deliberately did not build.

### "Your own table shows naive replay winning somewhere."

It does, and that row stays in. For `contrarian_sharp`, naive replay has the lowest error of the
four. That is the regime where the effective sample size is 1.62% and the tool refuses to answer
at all, so it is an estimator landing near the truth by accident in a case where nothing should
be trusted. The right reading is that the refusal is doing its job, and publishing the row is
cheaper than being caught omitting it.

### "How do I know the numbers in the README are current?"

Break one and watch the build fail. `docs/receipts-proof.md` is that transcript. The checker
matches phrases rather than digits, because searching for digits passes while the sentence
around them goes stale, and a reverse check fails on any number in the prose that no measurement
accounts for.

**If pushed on "who checks the checker":** three audits did, and between them found seven
holes, which is the honest headline about this mechanism. The anchor rule
required a phrase to appear somewhere in the checked set rather than in the document making the
claim, so a README whose every framing sentence had gone stale passed because the defense guide
still carried the wording. And fenced blocks were stripped before checking, which left the two
blocks of captured output on the README's first screen as the only unguarded numbers in the
document. Then a second pass found that the fix closed
mutation and left insertion open: guarding the lines that are there proves those lines are
current and says nothing about a fabricated line beside them, so a fabricated metric line inserted into
the headline block passed the build. And two metrics had been given the same anchor
phrase, so deleting one claim outright left the other satisfying the check on its behalf.

A third pass found two more. The insertion fix had been scoped to regions already containing a
percent sign, so a fabricated session count in a fence and a throughput claim in a code span
both passed, and a figure in image alt text passed too. And the anchor rule was per document,
which still allowed a sentence inverting the central finding, quoting the right number, to sit
a hundred lines from the sentence stating it correctly. The rule is now per occurrence: every
place a figure is written has to sit inside one of its anchor phrases or inside a guarded line.

The seventh was not in the receipts at all. `make experiments` had been broken since it was
written, the suite was at full line coverage and could not see it because nothing under `src/`
imports the runner, and the test I added to catch it ran the runner with `--help`, which
argparse answers before reaching any import. So the regression test passed against the
regression. That is this repository's own thesis happening to the repository, and I only found
out because somebody checked whether the check checked anything.

All seven are fixed, each of the receipts ones has a case in the transcript, which now runs ten
corruptions, and a test asserts that every case in it is a failure. The runner one is covered
by a test that reconstructs the broken runner and asserts the new check fails against it. The honest summary is that this mechanism was
wrong six times in ways I did not find myself. That is the argument for having something
adversarial read your work, and it is also why the transcript is committed rather than
described.

## Things to say, and things not to say

**Say:**

- "The deployed ranker replayed against its own log reports +5.53%, and its true lift over
  itself is zero by construction."
- "Two policies that rank identically and are served differently get the same replay estimate,
  to every digit, while their true values differ by 16.39%."
- "The tool refuses when the effective sample size falls below the floor, and exit code 2 means
  something different from exit code 1."
- "E5 found that the diagnostic moves the wrong way under propensity corruption, which is a
  limit of the design rather than a bug in it."
- "I predicted a symmetric failure and the measurement said one sided. I published the
  measurement."

**Do not say:**

- "Naive replay is always wrong." It is right, by accident, in the regime where the tool
  refuses, and that is in the README.
- "The corrected estimator fixes it." It calls a -0.05% policy a ship at +0.09%. Say "less
  wrong, and it reports when it cannot resolve the question".
- "Zero false positives" or any claim built on a measured zero. Nothing observed zero times in
  50,000 sessions is zero, it is below one in fifty thousand.
- "This works on real data." Say which claims transfer and which one needs a real log, as above.
- Any absolute timing. Nothing in this repository publishes one, because a published millisecond
  is a fact about the machine that ran it.

## The live demo, five commands

```bash
# 1. The log, before any candidate policy is involved.
replay-gap diagnose

# 2. The headline: the deployed ranker, replayed against the log it produced.
replay-gap evaluate incumbent --no-ranker

# 3. The refusal: the same ranker as contrarian_lift, served greedily. Exits 2.
replay-gap evaluate contrarian_sharp --no-ranker

# 4. Every policy against the exact referee, with the error against truth.
replay-gap compare --gap

# 5. Re-measure every published figure and check it against the prose.
make receipts
```

Run them in that order. Two is the hook, three is the product, four is the evidence, and five is
why any of it should be believed.
