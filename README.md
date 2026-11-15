# AI Automotive Studio

AI-powered application to isolate car images/videos, preserve the vehicle's appearance, and composite it into one fixed premium LED-lit automotive studio.

## Planned features
- JPG/JPEG/PNG image uploads; MP4/MOV video uploads
- Vehicle detection and segmentation
- Original background removal
- Fixed studio compositing
- Vehicle appearance preservation
- Fictional number-plate replacement
- Realistic shadows and floor reflections
- Consistent processing across video frames
- Optional multi-angle 360-degree viewer/spin video

A genuine 360-degree view requires source footage/images covering multiple angles. A single image cannot reveal unseen sides.

## Team ownership
| Person | Folder | Responsibilities |
|---|---|---|
| Aparna | `backend/` | FastAPI, uploads, AI pipeline orchestration, video extraction/reconstruction |
| Arya | `frontend/` | React interface, uploads, API calls, result presentation |
| You | `ai_engine/` | Detection, segmentation, background removal, fictional plate replacement, compositing |

## Development order
1. Agree on API and Python function contracts.
2. Validate still-image AI processing.
3. Connect backend image endpoint to frontend.
4. Add video frame processing and reconstruction.
5. Add optional 360-degree features after the core workflow works.

## Rules
- Never commit `.env`, passwords, API keys, credentials, private customer media, or generated videos.
- Keep large model weights out of ordinary Git commits; use approved external storage or Git LFS if needed.
- Work in feature branches and merge through reviewed pull requests.
- See `docs/` for individual assignments and integration workflow.
