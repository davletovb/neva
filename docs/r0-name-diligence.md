# R0 name and publication diligence (2026-10-19)

**Working name:** Hiyal / `hiyal`. Arabic *ḥiyal* (ingenious devices,
automata; also tricks), approximate pronunciation "hee-YAHL". The user has
already renamed the repository to
[`davletovb/hiyal`](https://github.com/davletovb/hiyal).

## Observations and risk

- Issue #103 reports that `hiyal` and several proposed subcrate names were
  unregistered on crates.io on 2026-10-19. This is an **issue-reported check**,
  not proof of current reservation or an immutable registry claim.
- [hiyal.com](https://hiyal.com/) is an **active, unrelated Uyghur-language
  computing resource** (including downloadable software and fonts); it is NOT
  available for Hiyal project branding. Do not redirect to it or claim it.
- Searches can miss unregistered rights, regional marks and similar spellings.
  A web search is not legal trademark clearance; authoritative searches for
  relevant jurisdictions and classes remain an external release prerequisite.
- GitHub user/org `hiyal` availability, registry ownership and any domain
  transfer have **not** been verified or acquired. The existing repository
  path is the canonical home.
- No crates.io tokens or authorized publisher account are available to this
  R0 change. No crate has been reserved or published by this PR.

## Binding decision

Keep **one unpublished package `hiyal`**, currently `publish = false`;
do not reserve a family of empty `hiyal-*` packages preemptively. Before the
first real publication, an authorized maintainer must (a) run a registry
ownership/name check with `cargo search hiyal --limit 20`, (b) complete
appropriate mark checks, (c) decide the permanent publication name, and (d)
publish a minimal **functional** documented crate with a real publisher
credential. If name rights cannot be cleared, change the crate name without
renaming the established GitHub project.

Publishing reserved placeholder packages would risk misleading users and is
not required to build or test R1. Name selection is resolved for development,
but final public publication authorization is **not** presumed complete.
