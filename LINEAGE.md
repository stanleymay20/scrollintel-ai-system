# ScrollIntel Repository Lineage

This document records the repository-family evidence used to identify `stanleymay20/scrollintel-ai-system` as the maintained ScrollIntel implementation and to classify earlier or empty sibling repositories safely.

## Canonical repository

`stanleymay20/scrollintel-ai-system`

Status: **CANONICAL / LINEAGE RESOLVED**.

The repository contains the developed ScrollIntel platform: multi-agent orchestration, backend and frontend application layers, data/ML workflows, deployment tooling, monitoring, testing infrastructure, and production-readiness documentation.

This lineage decision does **not** certify production readiness. The current repository still records open credential-hygiene remediation/closure criteria, and deployment health must be judged from current CI/security evidence rather than lineage status.

## `scroll-intel`

Status: **SUPERSEDED PROTOTYPE / HISTORY PRESERVED IN CANONICAL / SAFE ARCHIVE CANDIDATE**.

The prototype repository already declares itself superseded and points new development to `scrollintel-ai-system`.

More importantly, the last substantive prototype commit before its deprecation-only documentation commit is:

`06400f440697a4d63cf758e33197fc805be26145`

An exact Git comparison against canonical `main` proves that commit is the merge base and a direct ancestor of `scrollintel-ai-system`. At the time of this review, canonical `main` is 32 commits ahead and 0 behind that substantive prototype commit.

Therefore the prototype's meaningful implementation history is preserved inside the canonical Git graph. The later prototype-only commit `ce4c50fa9f8b6e79a2bc6e7574ea98a992c87391` merely marks the old repository as superseded and is not unique product implementation.

Decision: no new development should occur in `scroll-intel`. It may be archived rather than deleted when repository cleanup is performed.

## `ScrollIntel`

Status: **EMPTY PLACEHOLDER / SAFE ARCHIVE CANDIDATE**.

Repository metadata reports size `0`. No substantive implementation or controlled evidence was found during this family review.

## `scrollintel.`

Status: **EMPTY PLACEHOLDER / SAFE ARCHIVE CANDIDATE / NAMING DEBT**.

Repository metadata reports size `0`. No substantive implementation or controlled evidence was found. The trailing punctuation is inconsistent with the portfolio naming standard.

## Preservation and release rule

Repository consolidation is separate from production-readiness certification:

1. preserve canonical Git history, release evidence, security remediation records, CI evidence and deployment documentation;
2. archive rather than delete superseded or empty repositories unless a later review establishes a reason to remove them;
3. do not describe a project as production-ready merely because its lineage is resolved;
4. keep credential-rotation/revocation and historical-secret closure criteria explicit until independently verified;
5. treat current workflow results, tests and security evidence as the authority for release readiness.

## Current family decision

- `scrollintel-ai-system` — **CANONICAL / LINEAGE RESOLVED**.
- `scroll-intel` — **SUPERSEDED PROTOTYPE / HISTORY PRESERVED IN CANONICAL / SAFE ARCHIVE CANDIDATE**.
- `ScrollIntel` — **EMPTY PLACEHOLDER / SAFE ARCHIVE CANDIDATE**.
- `scrollintel.` — **EMPTY PLACEHOLDER / SAFE ARCHIVE CANDIDATE / NAMING DEBT**.

No repository has been archived or deleted by this decision.