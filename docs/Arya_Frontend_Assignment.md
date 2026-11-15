# Frontend Assignment — Arya

## Goal
Build a polished React demonstration interface. A user uploads a car image/video, submits it for processing, and views the actual result returned by the backend.

## Responsibilities
1. Create React app and responsive premium automotive-studio UI.
2. Support JPG/JPEG/PNG images and MP4/MOV videos.
3. Show selected-media previews and validate unsupported files.
4. Send uploads to Aparna's documented FastAPI endpoints.
5. Display loading, success, and error states based on actual API responses.
6. Display processed images and play processed videos; provide download/open action if result URL is available.
7. After core workflows work, add optional multi-angle 360-degree sequence viewing and spin-video playback.

A genuine 360-degree sequence requires source footage/images covering multiple angles; do not claim it from one image.

## Files to create
```text
frontend/
├── public/
├── src/
│   ├── assets/
│   ├── components/{UploadPanel.jsx,ImageResult.jsx,VideoResult.jsx,ProcessingStatus.jsx}
│   ├── pages/StudioPage.jsx
│   ├── services/api.js
│   ├── styles/global.css
│   ├── App.jsx
│   └── main.jsx
├── package.json
├── .env.example
└── README.md
```

## API agreement
Coordinate with Aparna on base URL, endpoint paths, multipart field name, immediate/asynchronous response, job status JSON, result URLs, and error format. Keep API calls in `src/services/api.js`.

## Work order
Create app/layout → image upload/preview → video upload/preview → agree API → connect image workflow → connect video workflow → refine UI → optional 360 viewer.

## Done when
Users can upload supported media, send it to the backend, and view actual processed results with correct loading/error handling.

## Boundaries
Arya owns browser UI. Aparna owns APIs/video processing. AI/ML owns model inference and image processing.
