# Development Workflow

## Branches
- `main` — stable integrated project
- `feature/backend-aparna` — Aparna
- `feature/frontend-arya` — Arya
- `feature/aiml` — AI/ML developer

## Rules
1. Work in your assigned component folder unless a shared change is agreed.
2. Pull latest `main` before starting.
3. Commit focused changes with clear messages.
4. Push to your feature branch.
5. Open a pull request into `main`; review and test before merging.
6. Never commit secrets, `.env`, virtual environments, `node_modules`, generated videos, or large model weights.

## Milestones
1. Agree on API and Python function contracts.
2. Validate AI image pipeline.
3. Connect backend image endpoint to AI pipeline.
4. Connect frontend image upload/result display.
5. Complete image workflow end-to-end.
6. Add video extraction, processing, and reconstruction.
7. Add optional 360-degree features.

## Shared contract checklist
Input formats/limits; RGB/BGR order; mask format/dimensions; AI function signatures/returns; API fields/JSON; job status/result URL; error format and cleanup rules.
