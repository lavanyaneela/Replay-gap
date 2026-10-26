# ADR-003: DuckDB by default, PostgreSQL compiled alongside, and why the split is stated

Status: accepted
Date: 2026-03-03

## Context

The log is queried with SQL rather than with pandas, because the questions asked of it (click
through rate by position, reward by logging propensity decile, how concentrated first slot
exposure is) are the questions a data engineer asks of a warehouse, and writing them as SQL is
what makes them legible to one.

That raises a dialect question, and a credibility question. A repository that claims both
engines and only ever runs one of them has made a claim it cannot support.

## Decision

Both dialects are hand written and committed under `sql/`. DuckDB executes on every run of
`make verify`, with no server and no network. PostgreSQL runs in an optional CI job against a
service container. The README and this ADR say which is which rather than implying both are
exercised everywhere.

## Why not transpile

The obvious labour saving move is to write the queries once and transpile. It was rejected
because the differences that matter here are not syntactic.

PostgreSQL truncates integer division; DuckDB does not. So this text:

```sql
SELECT SUM(clicked) / COUNT(*) AS ctr FROM impressions GROUP BY position
```

returns the right click through rate on DuckDB and exactly `0` for every position on
PostgreSQL, with no error on either. A transpiler preserves that faithfully, because it is valid
SQL in both dialects and means different things. The PostgreSQL file carries an explicit
`::numeric` cast, which is a decision somebody made rather than a conversion somebody automated.

The same applies to the median. DuckDB has `MEDIAN` as an ordinary aggregate; PostgreSQL offers
`PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ...)` and nothing else.

A test asserts that both directories expose the same query names and that each query selects the
same output columns under both dialects, which is the contract the Python side actually depends
on. Divergence in the SQL is allowed; divergence in the interface is not.

## Why DuckDB is the default

The repository promises that `make verify` passes from a fresh clone with no network. An
embedded engine keeps that promise. A server does not, and a repository whose headline command
needs Docker running is a repository most reviewers will not run.

## Why PostgreSQL is there at all

Writing a second dialect that nothing executes would be worse than writing one. The optional CI
job runs the PostgreSQL files against a real `postgres:16` service container, so the parity
claim has been tested by a machine rather than by inspection. The job is allowed to be skipped
on a fork with no container available, and when it is skipped the repository says the parity
check was structural on that run rather than pretending otherwise.

## The MLflow backend, recorded here because it is the same kind of decision

MLflow's filesystem tracking backend is refused outright by the current release, which raises
with a message saying it is in maintenance mode, and it never supported the model registry in
any release. So a repository that wants both tracking and a registry has three options: run a
tracking server, drop the registry, or use SQLite.

SQLite is one local file with no server and no network, so it satisfies the offline rule the
same way DuckDB does, and it supports registry operations. That is what ships. Registering a
policy is therefore a real operation that happens, not a comment describing one that would.

What gets registered is the policy's identity and the configuration that produced it, not a
serialised artefact, because a policy here is a scoring function. That is the honest content for
a registry in this design: its job is to record which version was evaluated against which log.

## Consequences

- `make verify` needs no container and no network.
- The dialect parity claim is backed by a structural test everywhere and by an executing test
  wherever a container is available.
- A reader wanting to run the PostgreSQL path locally needs Docker, and the CI workflow shows
  exactly how.
