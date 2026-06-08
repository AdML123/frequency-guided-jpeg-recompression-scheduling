# Security Policy

No credentials are required to inspect the included results, regenerate figures
or tables, or run smoke tests.

Do not commit account secrets, private data, model-store tokens, local
configuration files, or submission-system credentials. Report suspected leaks
privately to the repository owner.

Before publishing or uploading, run:

```powershell
conda run -n fgjpeg python scripts/check_no_secrets.py
```

GitHub upload should use GitHub CLI browser login, a personal access token kept
outside the repository, or SSH. Password authentication is not supported for Git
operations on GitHub.
