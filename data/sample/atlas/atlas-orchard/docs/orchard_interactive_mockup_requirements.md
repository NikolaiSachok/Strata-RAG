# Orchard — Interactive Mockup Requirements (RAW build instructions)

Instructions for a coding agent to build a React + Tailwind interactive mockup. This is a
build spec (routes, component props, state machines, mock data shapes), NOT product copy, so
the raw doc is EXCLUDED from embedding.

## Routes
- /            Home (tree view)
- /today       Today's list
- /harvest     Weekly basket summary

## Mock data
tasks: [{ id, title, done, dueAt }]
tree: { fruitCount, maxFruit, wilted }

## Interaction spec
Tapping a task toggles done and increments tree.fruitCount with a 300ms grow animation.
When fruitCount === maxFruit, navigate to /harvest and play the celebration sequence.
Every button, toggle, and modal must be wired; no dead controls.
