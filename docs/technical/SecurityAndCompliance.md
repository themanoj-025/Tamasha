# SecurityAndCompliance — Tamasha: Security & Compliance

| Field | Value |
| --- | --- |
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Security Engineer |
| Status | In Review |

---

## 1. Threat Model (STRIDE)

| Threat | Asset | Mitigation |
| --- | --- | --- |
| Spoofing | User identity | JWT + bcrypt/argon2; refresh rotation |
| Tampering | Video metadata | Role/ownership checks server-side |
| Repudiation | Publish/delete | Audit log with actor + timestamp |
| Info disclosure | Emails, watch history | TLS, encryption at rest, log masking |
| DoS | Feed/search | Rate limits + caching |
| Elevation | Creator/admin roles | Role middleware; admin separation |

## 2. Auth & Authz

- JWT access (15 min) + refresh (7 days), stored httpOnly where web.
- Role checks: creator required for publish/manage; ownership check per resource.
- Admin: reserved for platform operators; no self-service admin grant.

## 3. Data Classification

| Class | Examples | Handling |
| --- | --- | --- |
| PII | email, watch history, comments | TLS, encryption at rest, GDPR erase tool |
| Credentials | password hashes | Argon2/bcrypt, never plaintext |
| Public content | video metadata, stats | No restriction |
| Secrets | JWT secret, DB creds | Env vars / secret manager only |

## 4. Encryption Standards

- Transit: TLS 1.2+ at reverse proxy.
- At rest: AES-256 volume encryption for DB.
- Passwords: Argon2id (or bcrypt cost ≥ 12).

## 5. Compliance Checklist

- [ ] GDPR: account deletion → cascade watch/comment removal; hard delete on request
- [ ] Data minimization: stats aggregated, raw watch < 24 months
- [ ] Logging: no raw tokens/emails in logs
- [ ] Dependency scanning monthly
- [ ] Content moderation: flag + review workflow for videos

## 6. Incident Response (Outline)

1. Detect: alert on 5xx, auth failures, content flags.
2. Triage: identify affected surface (auth vs content vs infra).
3. Mitigate: revoke tokens, take down content, roll back deploy.
4. Recover: restore from backup; verify integrity.
5. Postmortem: within 48h in ../project/Tracker.md changelog.

## 7. Related Documents

| Document | Relationship |
| --- | --- |
| [API.md](API.md) | Auth endpoints |
| [Rules.md](../project/Rules.md) | Security baseline (Section 6) |
| [RiskRegister.md](../project/RiskRegister.md) | R-02, R-04, R-06 |
| [Schema.md](Schema.md) | PII tables |
