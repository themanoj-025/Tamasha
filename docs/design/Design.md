# Design — Tamasha: Design System & UX Principles

| Field | Value |
| --- | --- |
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Lead Designer |
| Status | In Review |

---

## 1. Design Principles

1. **Content-first** — thumbnails and titles lead; chrome recedes. *Do:* rich grid. *Don't:* heavy nav bars.
2. **Clarity over cleverness** — every action label is plain language.
3. **Fast by default** — skeleton loading, instant feedback, no dead clicks.
4. **Consistent hierarchy** — one headline size, one accent color.
5. **Trustworthy actions** — destructive ops require confirmation.

## 2. Brand & Visual Identity

- **Tone:** energetic, creative, professional. Content platforms speak through visuals.
- **Logo:** play/ticket motif; clear space 8px.
- **Imagery:** video thumbnails drive the page; UI uses minimal ornament.

## 3. Color System

| Token | Hex | Usage | Contrast |
| --- | --- | --- | --- |
| bg-canvas | #0F1115 | App background (dark) | — |
| bg-surface | #1A1E26 | Cards | — |
| text-primary | #F2F4F8 | Body | ≥ 7:1 |
| text-muted | #9AA3B2 | Secondary | ≥ 4.5:1 |
| accent | #FF4E6A | CTA, active states | ≥ 4.5:1 |
| success | #2EBD85 | Live/status | ≥ 4.5:1 |
| danger | #F25F5C | Errors, delete | ≥ 4.5:1 |
| border | #2A2F3A | Dividers | — |

## 4. Typography Scale

| Token | Font | Size | Weight | LH | Usage |
| --- | --- | --- | --- | --- | --- |
| display | Inter | 28px | 700 | 1.2 | Page hero |
| title | Inter | 20px | 600 | 1.3 | Section titles |
| body | Inter | 16px | 400 | 1.5 | Copy |
| caption | Inter | 13px | 400 | 1.4 | Metadata, timestamps |
| mono | JetBrains Mono | 14px | 400 | 1.5 | IDs, code |

## 5. Spacing & Grid

- Base 4px; scale 4/8/12/16/24/32/48/64.
- Video grid: 4 cols desktop, 2 tablet, 1 mobile; gutter 16px.
- Breakpoints: 640 / 1024 / 1440.

## 6. Component Library

### 6.1 Video Card

| State | Style |
| --- | --- |
| Default | Thumbnail 16:9, title 2-line clamp, meta row |
| Hover | Lift 4px, thumbnail scale 1.03 |
| Loading | Skeleton shimmer |

```
┌─────────────────┐
│   ████████████   │  thumbnail 16:9
├─────────────────┤
│ Title line 1    │
│ Title line 2    │
│ 👁 1.2K · 3 days│  meta row
└─────────────────┘
```

### 6.2 Buttons / Inputs / Toast

- Buttons: accent primary; ghost secondary; danger destructive; disabled 40% opacity.
- Inputs: 1px border, 2px accent focus ring; error state with message.
- Toast: slide-in 200ms, auto-dismiss 5s, left color border.

### 6.3 Publish Form

- Two-column layout: metadata fields + live preview panel.
- Save button shows progress; success → toast + redirect to detail.

## 7. Iconography & Imagery

- Stroke icons (Lucide-style), 20px default.
- Video/play, search, user, grid, plus, trash, edit, eye.

## 8. Accessibility

- WCAG 2.1 AA; full keyboard nav; skip-to-content link.
- aria-live for toasts; focus trap in modals.
- `prefers-reduced-motion` disables lift/scale transitions.

## 9. Responsive Behavior

| Breakpoint | Layout |
| --- | --- |
| < 640px | 1-col grid, bottom nav |
| 640–1024px | 2-col grid |
| > 1024px | 4-col grid, side nav |

## 10. Motion

| Token | Value |
| --- | --- |
| Duration | 150–250 ms |
| Easing | cubic-bezier(0.2,0,0,1) |
| Animated | card hover, toasts, modal fade |
| Never | color/verdict flips |

## 11. Dark Mode

- Default theme IS dark (bg-canvas #0F1115). Light mode deferred; token mapping reserved.

## 12. Related Documents

| Document | Relationship |
| --- | --- |
| [AppFlow.md](AppFlow.md) | Screens using these components |
| [PRD.md](../product/PRD.md) | UX requirements |
| [Rules.md](../project/Rules.md) | UI conventions for agents |
