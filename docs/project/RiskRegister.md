# RiskRegister — Tamasha: Known Risks & Mitigations

| Field | Value |
| --- | --- |
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Program Manager |
| Status | In Review |

| ID | Risk | Likelihood | Impact | Score | Mitigation | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-01 | Inappropriate content published | Medium | High | 9 | Flag + review workflow; creator terms | PM | Open |
| R-02 | Account takeover | Medium | High | 9 | Strong hashing, refresh rotation, rate-limited login | Sec | Open |
| R-03 | Search degrades with catalog size | Medium | Medium | 6 | FTS indexes, cache, ES upgrade path | Eng | Open |
| R-04 | Creator edits others' videos (IDOR) | Low | High | 6 | Server-side ownership checks + tests | Sec | Open |
| R-05 | Media provider outage | Medium | Medium | 6 | Health checks + fallback UI | DevOps | Open |
| R-06 | GDPR/PII breach | Low | High | 6 | Encryption, erase tool, log masking | Sec | Open |
| R-07 | Data loss on migration | Low | High | 6 | Backup policy + migration rollback | DevOps | Open |
| R-08 | Dup publishes from retries | Medium | Low | 3 | Idempotency keys | Eng | Open |

## Risk Matrix

```mermaid
quadrantChart
    title Risk Prioritization
    x-axis Low Likelihood --> High Likelihood
    y-axis Low Impact --> High Impact
    quadrant-1 Watch: R-03, R-05
    quadrant-2 Manage: R-08
    quadrant-3 Avoid: R-07, R-06, R-04
    quadrant-4 Critical: R-01, R-02
```

## Top 3 Focus Risks

1. **R-01 Content moderation** — flag + review before public exposure of flagged items.
2. **R-02 Account takeover** — enforce strong auth + rate limiting.
3. **R-07 Data loss** — nightly backups + tested restore.

## Related Documents

| Document | Relationship |
| --- | --- |
| [PRD.md](../product/PRD.md) | Top risk summary |
| [SecurityAndCompliance.md](../technical/SecurityAndCompliance.md) | Detail |
| [Tracker.md](Tracker.md) | Status updates |
