# Implementation Plan (agent-authored)

> NOTE: This is an agent-authored build/planning document. It describes engineering
> tasks, not product/theme content, and is NOISE for a content-focused corpus.

## Phase 1
- [ ] Scaffold Flutter project from template
- [ ] Wire up local SQLite persistence for habits
- [ ] Implement streak calculation service
- [ ] Add tide-chart widget (custom painter)

## Phase 2
- [ ] Integrate push notifications for reminders
- [ ] Add unit tests for the streak service
- [ ] Set up CI build

## Technical notes
Use Riverpod for state. Persist with drift. Target SDK 34.
