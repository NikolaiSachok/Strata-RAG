# Design System — coding-agent build instructions (RAW)

This file is a build spec consumed by a coding agent to generate the app's UI. It is NOT
product narrative — it is implementation detail (component inventory, token tables, spacing
scales). The raw doc is EXCLUDED from embedding; a future distillation track will embed a
compact design_intent summary instead.

## Color tokens
- color/primary: #F4C430
- color/surface: #FFF8E7
- color/danger: #C0563B

## Typography scale
- font/display: 32/40
- font/title: 24/32
- font/body: 16/24

## Components
Button (primary/secondary/ghost), Card, ListTile, ProgressRing, EmptyState, Toast.
Each component lists every variant, state (default/hover/pressed/disabled), and token binding.

## Spacing
4 / 8 / 12 / 16 / 24 / 32 dp scale. Grid gutter 16dp. Safe-area padding 24dp.
