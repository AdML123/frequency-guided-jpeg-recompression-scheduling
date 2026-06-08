from __future__ import annotations


def test_secret_scanner_flags_known_secret_patterns():
    from scripts.check_no_secrets import scan_text

    token = "ghp_" + "1234567890abcdefghij1234567890abcdef"
    password_word = "pass" + "word"
    findings = scan_text(
        f"token = {token} {password_word}='not-for-use'"
    )

    assert findings


def test_secret_scanner_flags_email_password_assignments():
    from scripts.check_no_secrets import scan_text

    password_word = "pass" + "word"
    findings = scan_text(f"email=user@example.test {password_word}='not-for-use'")

    assert "email_password_assignment" in findings
