"""
Video processing and orchestration service.
Handles video frame extraction, sequential AI processing with temporal smoothing,
audio preservation, video reconstruction, and asynchronous job tracking.
"""

import os
import time
import uuid
import shutil
import subprocess
import threading
from typing import Dict, Any, Optional
import cv2

from backend.app.core.config import settings
from backend.app.schemas.requests import VideoProcessOptions
from backend.app.schemas.responses import JobProgressResponse, JobStatus, VideoProcessResponse
from backend.app.services.file_service import FileService, CorruptedFileError
from backend.app.services.ai_pipeline_service import (
    ai_pipeline_service,
    NoVehicleDetectedException,
)


class JobManager:
    """Thread-safe in-memory job tracker for video and media processing."""

    def __init__(self):
        self._jobs: Dict[str, JobProgressResponse] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, kind: str = "video", total_frames: Optional[int] = None) -> JobProgressResponse:
        job = JobProgressResponse(
            job_id=job_id,
            kind=kind,
            status=JobStatus.PENDING,
            success=True,
            progress_pct=0.0,
            stage="initializing",
            current_frame=0,
            total_frames=total_frames,
            message="Job queued for processing.",
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_pct: Optional[float] = None,
        stage: Optional[str] = None,
        current_frame: Optional[int] = None,
        total_frames: Optional[int] = None,
        result_url: Optional[str] = None,
        download_url: Optional[str] = None,
        output_format: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[JobProgressResponse]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            if status is not None:
                job.status = status
                if status == JobStatus.FAILED:
                    job.success = False
                elif status == JobStatus.COMPLETED:
                    job.success = True
            if progress_pct is not None:
                job.progress_pct = round(progress_pct, 1)
            if stage is not None:
                job.stage = stage
            if current_frame is not None:
                job.current_frame = current_frame
            if total_frames is not None:
                job.total_frames = total_frames
            if result_url is not None:
                job.result_url = result_url
            if download_url is not None:
                job.download_url = download_url
            if output_format is not None:
                job.output_format = output_format
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if execution_time_ms is not None:
                job.execution_time_ms = execution_time_ms
            if metadata is not None:
                job.metadata = metadata

            return job

    def get_job(self, job_id: str) -> Optional[JobProgressResponse]:
        with self._lock:
            return self._jobs.get(job_id)


job_manager = JobManager()


class VideoService:
    """Orchestrates video frame extraction, AI processing, and video reconstruction."""

    @classmethod
    def _preserve_audio_if_feasible(cls, original_video_path: str, rendered_video_path: str, final_output_path: str) -> bool:
        """
        Attempt to extract audio from the original video and mux it into the rendered video using FFmpeg.
        Returns True if audio muxing succeeded; False if fallback is needed.
        """
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return False

        try:
            # Check if original video has an audio stream
            probe_cmd = [
                ffmpeg_bin,
                "-i", original_video_path,
                "-show_streams",
                "-select_streams", "a",
                "-loglevel", "error",
            ]
            probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if not probe_res.stdout and not probe_res.stderr:
                return False

            # Mux original audio into rendered video
            mux_cmd = [
                ffmpeg_bin,
                "-y",
                "-i", rendered_video_path,
                "-i", original_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                "-loglevel", "error",
                final_output_path,
            ]
            res = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            return res.returncode == 0 and os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0
        except Exception:
            return False

    @classmethod
    def process_video_pipeline(
        cls,
        job_id: str,
        input_video_path: str,
        original_filename: str,
        metadata: Dict[str, Any],
        options: VideoProcessOptions,
    ) -> JobProgressResponse:
        """
        Process frames from input_video_path, execute AI pipeline with temporal smoothing,
        and reconstruct playable video output.
        """
        start_time = time.perf_counter()
        job_dir = os.path.join(settings.TEMP_DIR, job_id)
        output_filename = f"studio_{job_id}.mp4"
        final_output_path = os.path.join(settings.RESULTS_DIR, output_filename)
        temp_rendered_path = os.path.join(job_dir, "rendered_raw.mp4")

        total_frames = metadata.get("total_frames", 1)
        fps = metadata.get("fps", 24.0)

        job_manager.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress_pct=2.0,
            stage="initializing",
            message="Extracting video frames and initializing vision pipeline...",
        )

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            err_msg = "Could not open video file for processing."
            job_manager.update_job(job_id, status=JobStatus.FAILED, stage="failed", error=err_msg, message=err_msg)
            FileService.cleanup_job_workspace(job_id)
            raise CorruptedFileError(err_msg)

        # Use approved studio dimensions (1920x1080) for composited studio video
        out_w, out_h = 1920, 1080
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_rendered_path, fourcc, fps, (out_w, out_h))

        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(temp_rendered_path, fourcc, fps, (out_w, out_h))

        if not writer.isOpened():
            cap.release()
            err_msg = "Failed to initialize video encoder on the host system."
            job_manager.update_job(job_id, status=JobStatus.FAILED, stage="failed", error=err_msg, message=err_msg)
            FileService.cleanup_job_workspace(job_id)
            raise RuntimeError(err_msg)

        state: Dict[str, Any] = {}
        frame_idx = 0
        detected_frames_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1

                # Process frame through AI pipeline with inter-frame temporal smoothing
                try:
                    comp_frame, frame_meta = ai_pipeline_service.process_video_frame(
                        frame=frame,
                        state=state,
                        fictional_plate_text=options.fictional_plate_text,
                        enable_temporal_smoothing=options.enable_temporal_smoothing,
                    )
                    if frame_meta.get("has_detection", False):
                        detected_frames_count += 1
                except Exception as frame_err:
                    # In case of rare frame-level error, resize to studio dimensions to maintain continuity
                    comp_frame = cv2.resize(frame, (out_w, out_h))

                writer.write(comp_frame)

                # Periodic progress update scaled between 5% and 92%
                progress = round(5.0 + (frame_idx / max(total_frames, 1)) * 87.0, 1)
                job_manager.update_job(
                    job_id,
                    progress_pct=min(progress, 92.0),
                    stage="processing_frames",
                    current_frame=frame_idx,
                    message=f"Detecting & segmenting vehicles (frame {frame_idx} of {total_frames})",
                )

            cap.release()
            writer.release()

            # Check if any vehicle was detected at all throughout the video
            if detected_frames_count == 0 and frame_idx > 0:
                err_msg = "No vehicle was detected in the provided video."
                job_manager.update_job(job_id, status=JobStatus.FAILED, stage="failed", error=err_msg, message=err_msg)
                raise NoVehicleDetectedException(err_msg)

            # Audio preservation and final video encoding step
            job_manager.update_job(
                job_id,
                progress_pct=95.0,
                stage="rendering",
                message="Compositing and encoding studio video...",
            )
            audio_muxed = cls._preserve_audio_if_feasible(input_video_path, temp_rendered_path, final_output_path)
            if not audio_muxed:
                # Direct move of rendered video if ffmpeg not present or audio not muxed
                if os.path.exists(final_output_path):
                    os.remove(final_output_path)
                shutil.move(temp_rendered_path, final_output_path)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            result_url = f"/api/results/{output_filename}"
            download_url = f"/api/results/{output_filename}?download=true"

            job_resp = job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress_pct=100.0,
                stage="completed",
                current_frame=frame_idx,
                total_frames=frame_idx,
                result_url=result_url,
                download_url=download_url,
                output_format="mp4",
                message="Video processing completed successfully.",
                execution_time_ms=elapsed_ms,
                metadata={
                    "original_filename": original_filename,
                    "fps": fps,
                    "frame_count": frame_idx,
                    "dimensions": {"width": out_w, "height": out_h},
                    "detected_frames": detected_frames_count,
                    "audio_preserved": audio_muxed,
                },
            )
            return job_resp

        except Exception as e:
            if 'cap' in locals() and cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if 'writer' in locals() and writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            job_manager.update_job(job_id, status=JobStatus.FAILED, stage="failed", error=str(e), message=f"Processing failed: {e}")
            if not options.async_mode:
                raise e
            return job_manager.get_job(job_id)
        finally:
            FileService.cleanup_job_workspace(job_id)

    @classmethod
    def start_video_processing(
        cls,
        file_path: str,
        original_filename: str,
        metadata: Dict[str, Any],
        options: VideoProcessOptions,
        job_id: str,
    ) -> JobProgressResponse:
        """
        Initiate video processing either synchronously or asynchronously based on options.async_mode.
        """
        if options.async_mode:
            # Run in worker thread
            thread = threading.Thread(
                target=cls.process_video_pipeline,
                args=(job_id, file_path, original_filename, metadata, options),
                daemon=True,
            )
            thread.start()
            return job_manager.get_job(job_id)
        else:
            # Synchronous processing (preferred by Arya's frontend)
            return cls.process_video_pipeline(job_id, file_path, original_filename, metadata, options)


video_service = VideoService()
