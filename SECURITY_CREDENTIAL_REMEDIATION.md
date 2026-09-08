# ScrollIntel Credential Remediation

## Status

A repository hygiene review on 8 September 2026 found multiple non-template environment files tracked on the public `main` branch.

The branch tip has been remediated by:

- removing the tracked non-template environment files;
- retaining only explicit `.example` and `.template` environment files;
- hardening `.gitignore` so `.env` and `.env.*` files are ignored by default;
- preserving explicit exceptions only for example/template files;
- removing obsolete README/documentation claims that positioned the system as replacing accountable technical leadership.

## Important limitation

Removing a secret-bearing file from the current branch tip **does not remove it from Git history** and does not revoke any credential that may have appeared in that file.

Therefore this remediation is not considered closed until historical credential exposure has been assessed.

## Required closure steps

1. Identify every credential class that may have existed in historical environment files.
2. Treat any real credential that was committed to a public repository as potentially exposed.
3. Rotate or revoke affected credentials at the relevant provider.
4. Confirm applications and deployment environments use the new credentials through secret stores or deployment settings rather than tracked files.
5. Search the repository history for additional secret-bearing files or plaintext credentials.
6. Rewrite Git history where appropriate after rotation, understanding that history rewriting cannot guarantee removal from existing clones, caches or forks.
7. Re-run secret scanning and verify no active credentials remain in the repository.
8. Record the date and evidence of credential rotation without recording secret values.

## Repository policy going forward

- Secrets must be provided through local environment files, CI/CD secret stores or deployment-provider secret management.
- Real `.env` files must never be committed.
- Example files may contain variable names and clearly non-secret placeholder values only.
- Production readiness must not be claimed solely because environment files have been removed from the latest commit.

## Current portfolio status

Until the historical credential review and any necessary rotations are complete, ScrollIntel should be treated as an engineering project undergoing security hardening rather than a production-certified system.
