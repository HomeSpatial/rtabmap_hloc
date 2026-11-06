# Security reporting

This alpha project has no guaranteed security-support window. Fixes target the
current development version. Avoid publishing credentials, private photographs
or complete databases in a public issue.

Use GitHub's private vulnerability reporting option if it is enabled for this
repository. If it is unavailable, open an issue requesting a private contact
without posting vulnerability details or sensitive reproductions. Maintainers
must establish a private channel before receiving sensitive material.

Treat databases, image decoders and model downloads as external inputs. Run
untrusted material in an isolated environment with appropriate resource limits.
HLOC can download executable model code through upstream loaders; the Python
lockfile alone is not a complete runtime model supply-chain lock.

Include the package version/commit, environment, affected input format, expected
versus observed behavior and a synthetic reproducer where possible.
