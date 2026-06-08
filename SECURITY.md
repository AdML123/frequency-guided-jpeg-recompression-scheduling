# Security Policy

This artifact does not require credentials, private datasets, model-store
tokens, or submission-system accounts for smoke tests, figures, tables, or the
runtime microbenchmark.

Do not open a public issue containing secrets. Report suspected leaks privately
to the repository owner. Before publishing or uploading, run:

```powershell
conda run -n paper20-cu128 python scripts/check_no_secrets.py
```

Safe GitHub upload must use GitHub CLI browser login with
`gh auth login --web`, a personal access token stored outside the repository, or
SSH. Password authentication is not supported for Git operations on GitHub. Do
not include real credentials in repository files, commits, remotes, issues, or
workflow logs.
