# Security Policy

thinchat's core purpose is handling provider API keys without leaking them: it redacts the key
from error messages, severs the exception chain so a key-bearing SDK error is unreachable from
what a caller catches, and pins each provider's official endpoint so an environment-controlled
base URL cannot redirect a resolved key. If you find a way a key can still escape — in an error,
a traceback, a log line, or a redirected request — please report it privately rather than opening
a public issue.

## Reporting a vulnerability

- Use GitHub's private vulnerability reporting for this repository (**Security → Report a
  vulnerability**), or email seokhoonj@gmail.com.
- Please do not paste a real API key into a report — a redacted example or the code path is enough.

## Supported versions

The latest released 0.x version receives security fixes.
