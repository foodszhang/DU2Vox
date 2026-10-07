# Research workflow

This directory is the canonical project-state layer shared across interactive web reasoning and local coding agents.

## Source-of-truth order

1. Reproducible repository evidence and run artifacts
2. `METHOD_SPEC.md` and accepted `DECISIONS.md`
3. The active task packet
4. `CLAIMS.md`
5. Paper prose and figures
6. Chat history or model memory

## Task execution

For a bounded research task:

1. Read the task packet.
2. Read only the state files referenced by the task, plus this workflow.
3. Preserve frozen constraints.
4. Record code/config/data provenance.
5. Produce a result packet under `results/`.
6. Do not promote or rewrite scientific claims unless explicitly authorized.

## Scientific ambiguity

If execution exposes an ambiguity in the method or claim, stop that branch of interpretation and record the ambiguity. Do not invent a scientific resolution in code, figures, or prose.
