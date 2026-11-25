# YouTube Splat AI

Prototype for reconstructing static scenes from authorized YouTube videos. React dashboard, FastAPI job service, Nerfstudio Splatfacto and experimental TensorFlow/Keras Gaussian renderer. The latter uses 512 isotropic Gaussians and 48x48 training images; it is not equivalent to full 3DGS. Both export PLY; other formats are not implemented.

## Local GPU
Linux, NVIDIA GPU, compatible CUDA driver and NVIDIA Container Toolkit required. Install Docker Compose. Run `docker compose up --build` from the root. For the optional Keras engine use `INSTALL_KERAS=true docker compose up --build`; TensorFlow and PyTorch CUDA dependencies may need separate environments if conflicts arise. Alternatively install dependencies in Python 3.10 environments, install FFmpeg and COLMAP, and run `python -m uvicorn backend.main:app --port 8000`. The Keras subprocess uses the same Python environment by default.

In another terminal: `cd frontend && npm install && npm run dev`. Open http://localhost:5173. Only submit videos you are authorized to download. Default video is an example URL, not a verified reconstruction.

## Cloud GPU / Vercel
Build the same Docker image on a GPU VM or container service with persistent /data storage. Expose port 8000 through an authenticated HTTPS reverse proxy. Set ORIGINS to the exact frontend origin. In Vercel import the repository, set root directory to frontend and VITE_API_URL to the HTTPS worker endpoint. Rebuild after changing the URL. Do not deploy the Python training worker as a Vercel function. No cloud resources are provisioned by this project.

## API
POST /jobs accepts url, engine (gsplat or keras), frames and iterations. GET /jobs/{id} returns status, stage and latest log line, plus model URL on completion. GET /health checks API availability. Jobs are serialized in a single process; do not run multiple uvicorn workers. Restart recovery, authentication, quotas, cancellation, durable queues and full production hardening are not implemented. Do not expose the unauthenticated API publicly. The dashboard polls every two seconds. Training progress is log-based rather than a guaranteed percentage.

## Tests
`python -m pytest -q` runs URL validation tests. `python -m compileall backend` checks syntax. Full GPU training, Docker builds and browser rendering require separate integration verification. COLMAP requires sufficient overlapping viewpoints and a mostly static scene; YouTube access may fail due to platform restrictions.

## Upload a video from your device
Select **Upload from device** in the Video source menu, browse for an MP4, MOV, AVI, WebM, MKV or M4V file (maximum 500 MiB), select the engine and quality, and generate. The browser sends multipart data directly to the GPU backend at POST /jobs/upload; uploaded videos skip YouTube ingestion and use the same reconstruction pipeline. Extension and size are checked; decoding is performed during reconstruction. Configure reverse proxy body limits and authentication before public use. Uploaded files persist on the worker volume; only exported PLY files are served by the model endpoint. Upload tests mock the training worker and do not verify real video decoding.

## Production frontend
https://youtube-splat-ai.vercel.app

Use the Vercel Vite project with root directory `frontend`, build command `npm run build`, output `dist`. Do not select FastAPI for this project. The GPU service entrypoint is `backend.main:app` and runs via Docker/Uvicorn outside Vercel. Configure `VITE_API_URL` at build time or enter the HTTPS GPU worker URL in the dashboard and test `/health`. Set worker `ORIGINS=https://youtube-splat-ai.vercel.app,https://youtube-splat-ai-acme-fdd1.vercel.app`. Production does not default to localhost.

## Local full-stack mode (Windows WSL2 / Linux)
Install Docker Desktop with WSL2 GPU support on Windows, or Docker Engine and NVIDIA Container Toolkit on Linux. A compatible NVIDIA GPU and driver are required. From the repository root run `docker compose up --build` to start both services. Open http://localhost:5173, select **Use local backend**, and click **Test connection**. API health: http://localhost:8000/health. Both ports bind to loopback only. Stop with `docker compose down`; data persists in ./data. To enable experimental Keras, set INSTALL_KERAS=true before building (PowerShell: `$env:INSTALL_KERAS='true'`). The Vercel frontend remains remote-only; use the local frontend for HTTP loopback access. Without Docker, start Uvicorn with `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` in a prepared GPU Python environment, then run `npm install` and `npm run dev -- --strictPort` inside frontend in another terminal. Local GPU builds and training require hardware verification.

## Troubleshooting headless COLMAP
If feature extraction aborts with Qt xcb or cv2/qt/plugins errors, rebuild the worker (`docker compose build --no-cache worker`, then `docker compose up -d worker`). The image isolates COLMAP's Qt plugins from OpenCV and selects offscreen Qt. If GPU SIFT still fails (e.g. OpenGL context), set `COLMAP_CPU=true` (PowerShell: `$env:COLMAP_CPU='true'`) and run `docker compose up -d --force-recreate worker`. This selects CPU COLMAP preprocessing only; gsplat training still requires CUDA. Submit a new job and inspect `data/<job-id>/training.log` if it fails. Existing failed jobs are not resumed automatically.
