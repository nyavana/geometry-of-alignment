## MODIFIED Requirements

### Requirement: Presentation slides
The slide deck and paper writeup SHALL be produced by an agent (or small agent team) working on the `agent/writeup` branch, AFTER the human verification gate (M4) has passed, and SHALL cite only experimental numbers that exist in files under `results/` with a verifiable file path and commit hash.

#### Scenario: Slide deck complete
- **WHEN** the presentation is prepared
- **THEN** it SHALL contain the hiking emergency scenario as the opening, at least 3 key result figures, and a course-connections section

#### Scenario: Writeup gated by human verification
- **WHEN** the writeup agent begins drafting paper prose or slide bullets
- **THEN** `STATUS_FOR_HUMAN.md` SHALL already exist on the `agent/writeup` branch AND contain the operator's green-light sentence, otherwise the writeup agent SHALL stop and report the gate as unresolved

#### Scenario: All numeric claims traceable
- **WHEN** any paper section or slide quotes a refusal rate, signal strength, alpha value, or Frobenius norm
- **THEN** the same number SHALL appear in a file under `results/`, and the writeup agent SHALL record the source path and commit hash either inline or in a companion `paper/sources.md` file
