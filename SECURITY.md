# Security policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in this project, please **do not open a public issue**. Email **security@openie.dev** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any suggested mitigation or fix

You should receive an acknowledgement within **3 business days**, and a substantive response within **10 business days**.

We follow coordinated disclosure: please give us **90 days** to patch and release a fix before public disclosure.

## Supported versions

Only the latest released version of each binary receives security updates. We do not backport fixes to older versions.

| Version | Supported |
|---|---|
| Latest release | ✓ |
| Older releases | ✗ |

## Out of scope

- Bugs in unsupported (older) versions
- Theoretical attacks without working proof-of-concept
- Issues in dependencies that don't have a security impact on this project
- Self-XSS, missing security headers without demonstrated impact
