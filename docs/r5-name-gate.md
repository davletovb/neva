# Hiyal name diligence — 20 October 2026

**Decision:** development project name and Rust package remain **Hiyal** /
`hiyal` (Arabic *ḥiyal*, pronounced approximately "hee-YAHL",
"ingenious devices"/automata; the term also has other historical meanings).
Canonical code repository: `davletovb/hiyal`.

This confirms the **selection and documentation** requirement, not ownership
of a namespace, registration of a trademark or successful public publication.

## Evidence and collisions

- Historical 19 October search in [issue #103](https://github.com/davletovb/hiyal/issues/103)
  found no crates.io listing for `hiyal` or the proposed subcrate names **on
  that date**. A new name must be checked directly against the crates.io API
  and its current owners before publication; no vacancy is guaranteed today.
- Public search on 20 October did not produce an authoritative registry
  registration result for `hiyal`. That is **not proof** the name is free.
  A distinct unrelated project uses "Hiyal" for a software hypervisor
  (see [public issue](https://github.com/swe-bench/SWE-bench/issues/586)).
  Naming confusion and rights should be reassessed before a launch.
- [hiyal.com](https://hiyal.com/) belongs to an unrelated Uyghur-language
  computing resource; it is not a project asset. No GitHub `hiyal`
  username/organization or domain is claimed.
- Formal searches in the relevant Nice classes and jurisdictions remain
  required, e.g. [WIPO Global Brand Database](https://www.wipo.int/en/web/global-brand-database/index)
  and [USPTO Trademark Search](https://tmsearch.uspto.gov/). Internet search
  is not a legal clearance opinion. No trademark filing has been made here.
- The Rust package keeps `publish = false`. There is no verified crates.io
  publisher account, token, package ownership, reservation or external
  release authorization; no package was published.

## Release checklist — distinct from selecting a development name

- [x] Select and document the name, pronunciation and canonical repository
- [x] Record known naming collisions and the unowned domain
- [ ] Complete live registry/owner check immediately before publication
- [ ] Complete jurisdiction-appropriate similarity/trademark clearance
- [ ] Confirm maintainer publishing ownership and authorize publication
- [ ] Publish a functional package (no misleading placeholder reservation)
- [ ] Verify crate installation and public reproducible artifact provenance

Those open **external publication gates** must not be silently reinterpreted
as a failed local Rust test. They must not block private native-development
work, but they do block any claim that a public release is ready.
