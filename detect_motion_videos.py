"""
Tesla Sentry Mode Motion Detection Tool

This script analyzes Tesla Sentry Mode footage to identify videos containing significant motion.
It creates an interactive HTML report with thumbnails and provides real-time updates during
processing.

Key features:
- Motion detection with adjustable thresholds
- GPU acceleration using CUDA when available
- Multi-threaded parallel processing
- Interactive HTML reports with status updates and auto-refresh toggle
- Video thumbnails and filtering options
- Camera type detection and filtering

Copyright (c) 2025 Ariel Jacob
License: MIT - See LICENSE file for details
"""

import os
import cv2
import argparse
import numpy as np
import time
import datetime
import sys
import re
import base64
import logging
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathlib  # Use pathlib instead of os.path
from collections import defaultdict

# Set up logging to file only
log_file = "motion_detection_log.txt"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Set the logger to not propagate to the root logger (which outputs to console)
logger.propagate = False

# Try to import rich, but provide fallback if not available
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.columns import Columns
    USE_RICH = True
except ImportError:
    USE_RICH = False
    print("Rich library not found - using basic terminal output")

# Global abort flag for graceful termination
abort_requested = False
abort_file = "abort.txt"
abort_check_interval = 1.0  # Check for abort file every second

# Function to check for abort file
def check_for_abort_file():
    global abort_requested
    while not abort_requested:
        if os.path.exists(abort_file):
            logger.info(f"Abort file '{abort_file}' detected. Initiating graceful shutdown.")
            abort_requested = True
            # Print message to console
            print(f"\n[{get_timestamp()}] Abort file detected. Stopping all workers. Please wait for graceful shutdown...")
            break
        time.sleep(abort_check_interval)

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    global abort_requested
    if not abort_requested:
        logger.info("Keyboard interrupt (Ctrl+C) detected. Initiating graceful shutdown.")
        abort_requested = True
        print(f"\n[{get_timestamp()}] Keyboard interrupt received. Stopping all workers. Please wait for graceful shutdown...")
    else:
        print(f"\n[{get_timestamp()}] Second interrupt received. Forcing immediate exit.")
        sys.exit(1)

def has_cuda_support():
    """Check if OpenCV is built with CUDA support and a compatible GPU is available."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except:
        return False

def detect_motion(video_path, threshold=45, min_area_percentage=2.5, sample_frames=True, use_gpu=True, 
                 video_index=0, total_videos=1, worker_id=0, progress_updater=None, html_writer=None):
    """
    Detect substantial motion in a video file.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for motion detection (0-255)
        min_area_percentage: Minimum percentage of the frame that needs to show motion
        sample_frames: If True, sample frames instead of processing all frames
        use_gpu: If True, use GPU acceleration if available
        video_index: Index of this video in the total list
        total_videos: Total number of videos to process
        worker_id: ID of the worker processing this video
        progress_updater: Function to call to update progress
        html_writer: HTMLProgressWriter instance to update HTML in real-time
    
    Returns:
        True if substantial motion is detected, False otherwise
    """
    try:
        global abort_requested
        filename = os.path.basename(video_path)
        start_time = time.time()
        
        logger.info(f"Worker {worker_id}: Starting processing of {filename}")
        
        if progress_updater:
            progress_updater(worker_id, f"Starting: {filename}", video_index)
        
        # Check if abort was requested before we start
        if abort_requested:
            logger.info(f"Worker {worker_id}: Abort requested, skipping {filename}")
            if progress_updater:
                progress_updater(worker_id, f"Aborted: {filename}", video_index)
            return False
        
        # Check if GPU acceleration is available and requested
        cuda_available = has_cuda_support()
        use_cuda = use_gpu and cuda_available
        
        if use_cuda:
            logger.info(f"Worker {worker_id}: Using CUDA GPU acceleration for {filename}")
        else:
            logger.info(f"Worker {worker_id}: Using CPU processing for {filename}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Worker {worker_id}: Failed to open video file {filename}")
            if progress_updater:
                progress_updater(worker_id, f"Error opening: {filename}", video_index)
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Worker {worker_id}: Video {filename} properties: {frame_width}x{frame_height}, {frame_count} frames, {fps} fps")
        
        if progress_updater:
            progress_updater(worker_id, f"Processing: {filename} - {frame_width}x{frame_height}, {frame_count} frames", video_index)
        
        if frame_count < 10:  # Very short videos may not have enough frames
            logger.warning(f"Worker {worker_id}: Video {filename} is too short ({frame_count} frames), skipping")
            if progress_updater:
                progress_updater(worker_id, f"Skipping (too short): {filename}", video_index)
            cap.release()
            return False
        
        # Calculate the minimum motion area in pixels
        min_motion_area = (frame_width * frame_height) * (min_area_percentage / 100)
        logger.info(f"Worker {worker_id}: Minimum motion area for {filename}: {min_motion_area} pixels ({min_area_percentage}% of frame)")
        
        # Initialize the first frame
        ret, prev_frame = cap.read()
        if not ret:
            logger.error(f"Worker {worker_id}: Failed to read first frame from {filename}")
            if progress_updater:
                progress_updater(worker_id, f"Error reading first frame: {filename}", video_index)
            cap.release()
            return False
        
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        
        # Initialize GPU mats if CUDA is enabled
        if use_cuda:
            gpu_prev = cv2.cuda_GpuMat()
            gpu_prev.upload(prev_frame)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0)
        
        # Sample interval for processing frames
        sample_interval = 10 if sample_frames else 1
        total_iterations = frame_count // sample_interval
        
        # Update progress every 2 seconds or 100 frames, whichever comes first
        last_update_time = time.time()
        update_interval = 2.0  # seconds
        frame_update_interval = 100 // sample_interval  # frames
        
        logger.info(f"Worker {worker_id}: Processing {filename} with sample interval {sample_interval}, total iterations: {total_iterations}")
        
        # Process frames
        for i in range(1, frame_count, sample_interval):
            # Check for abort request
            if abort_requested:
                logger.info(f"Worker {worker_id}: Abort requested during processing of {filename}")
                if progress_updater:
                    progress_updater(worker_id, f"Aborted: {filename}", video_index)
                cap.release()
                return False
            
            # Set the frame position
            if sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Worker {worker_id}: Failed to read frame {i} from {filename}")
                break
                
            # Update progress periodically based on time or frame count
            current_time = time.time()
            time_to_update = (current_time - last_update_time) >= update_interval
            frame_to_update = (i % (sample_interval * frame_update_interval) == 0) or (i == 1)
            
            if time_to_update or frame_to_update:
                # Just update with the current frame number - no percentage needed
                update_msg = f"Processing: {filename} - frame {i}/{frame_count}"
                logger.info(f"Worker {worker_id}: {update_msg}")
                if progress_updater:
                    progress_updater(worker_id, update_msg, video_index)
                last_update_time = current_time
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if use_cuda:
                # GPU-accelerated processing
                gpu_frame.upload(gray)
                
                # Apply Gaussian blur
                blurred = cv2.cuda_GpuMat()
                gpu_filter.apply(gpu_frame, blurred)
                
                # Compute the absolute difference between the current and previous frame
                gpu_delta = cv2.cuda.absdiff(gpu_prev, blurred)
                
                # Apply threshold
                gpu_thresh = cv2.cuda.threshold(gpu_delta, threshold, 255, cv2.THRESH_BINARY)[1]
                
                # Download result to CPU for dilation and analysis
                thresh = gpu_thresh.download()
                
                # Dilate the thresholded image to fill in holes
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Update the previous frame
                gpu_prev = blurred.clone()
            else:
                # CPU processing
                # Apply Gaussian blur
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # Compute the absolute difference between the current and previous frame
                frame_delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
                
                # Dilate the thresholded image to fill in holes
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Update the previous frame
                prev_frame = gray
            
            # Calculate the percentage of the image with motion
            motion_pixels = np.sum(thresh == 255)
            
            # Log motion pixel count for verbose debugging
            if logger.level == logging.DEBUG:
                logger.debug(f"Worker {worker_id}: {filename} frame {i} - motion pixels: {motion_pixels}/{min_motion_area}")
            
            # If motion area is larger than minimum area, we have substantial motion
            if motion_pixels > min_motion_area:
                elapsed_time = time.time() - start_time
                logger.info(f"Worker {worker_id}: MOTION DETECTED in {filename} at frame {i} - {motion_pixels} pixels exceed threshold (min: {min_motion_area})")
                if progress_updater:
                    progress_updater(worker_id, f"MOTION DETECTED: {filename} - {elapsed_time:.1f}s", video_index)
                
                # If we have an HTML writer, add the video to it immediately
                if html_writer:
                    html_writer.add_video(video_path)
                    
                cap.release()
                return True
        
        elapsed_time = time.time() - start_time
        logger.info(f"Worker {worker_id}: No significant motion detected in {filename} after {elapsed_time:.1f}s")
        if progress_updater:
            progress_updater(worker_id, f"No motion: {filename} - {elapsed_time:.1f}s", video_index)
        cap.release()
        return False
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error processing {filename}: {str(e)}", exc_info=True)
        if progress_updater:
            progress_updater(worker_id, f"Error: {filename} - {str(e)}", video_index)
        return False

def get_timestamp():
    """Return current timestamp for progress messages"""
    return datetime.datetime.now().strftime("%H:%M:%S")

class SimpleProgressTracker:
    """Fallback progress tracker when rich is not available"""
    
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.worker_status = ["Idle" for _ in range(num_workers)]
        self.start_time = time.time()
        self.completed = 0
        self.motion_found = 0
        self.total_videos = 0
        self.video_progress = {}  # Track progress of each video
        
        # Print header
        print("\n==== Motion Detection Progress ====")
        print(f"Workers: {num_workers}")
    
    def set_total_videos(self, total):
        """Set the total number of videos to process"""
        self.total_videos = total
        print(f"Total videos to process: {total}")
    
    def update_worker_status(self, worker_id, status, video_index, percent_done=None):
        """Update status for a specific worker"""
        if worker_id < len(self.worker_status):
            # Truncate status if too long
            if len(status) > 80:
                status = status[:77] + "..."
                
            self.worker_status[worker_id] = status
            
            # Track video progress
            if percent_done is not None:
                self.video_progress[video_index] = percent_done
            
            # Print certain status changes
            if "MOTION DETECTED" in status:
                print(f"Worker {worker_id+1}: {status}")
                self.motion_found += 1
                self.completed += 1
                self._print_status_update()
            elif "Error" in status:
                print(f"Worker {worker_id+1}: {status}")
            elif "No motion" in status:
                print(f"Worker {worker_id+1}: {status}")
                self.completed += 1
                self._print_status_update()
            elif not ("Processing" in status and "%" in status):
                # Don't print every progress update to avoid spam
                print(f"Worker {worker_id+1}: {status}")
    
    def _print_status_update(self):
        """Print a status update when a video is completed"""
        elapsed = time.time() - self.start_time
        
        if self.completed > 0:
            avg_time_per_video = elapsed / self.completed
            remaining_videos = self.total_videos - self.completed
            estimated_remaining = avg_time_per_video * remaining_videos
        else:
            estimated_remaining = 0
        
        percent = (self.completed / self.total_videos * 100) if self.total_videos > 0 else 0
        
        print(f"\n[{get_timestamp()}] Progress: {self.completed}/{self.total_videos} videos ({percent:.1f}%)")
        print(f"Motion detected: {self.motion_found} videos")
        print(f"Time elapsed: {format_time(elapsed)}")
        print(f"Estimated remaining: {format_time(estimated_remaining)}\n")
    
    def final_summary(self, motion_videos):
        """Display final summary after processing is complete"""
        print(f"\n==== Processing Complete! ====")
        print(f"Total videos analyzed: {self.total_videos}")
        print(f"Videos with motion detected: {len(motion_videos)}")
        print(f"Total time: {format_time(time.time() - self.start_time)}")
        
        if motion_videos:
            print(f"\nVideos with substantial motion:")
            for i, video in enumerate(motion_videos):
                print(f"{i+1}. {video}")

class RichProgressTracker:
    """Class to track and display progress using rich library"""
    
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.worker_status = ["Idle" for _ in range(num_workers)]
        self.start_time = time.time()
        self.completed = 0
        self.motion_found = 0
        self.total_videos = 0
        self.video_progress = {}  # Track progress of each video
        self.refresh_rate = 0.25   # Increase refresh rate - update display every quarter second instead of half second
        self.last_refresh = time.time()
        
        # Initialize rich components
        self.console = Console()
        
        # Create progress display with spinner instead of percentage
        # Increase the width of the status column and reduce space before time
        self.progress = Progress(
            TextColumn("[bold blue]{task.description:<12}"),  # Fixed width worker column
            TextColumn("{task.fields[spinner]:>2}"),  # Centered spinner with fixed width
            TextColumn(" • "),  # Separator
            TextColumn("[bold]{task.fields[status]:<100}"),  # Wider status column with left alignment
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=8,  # Higher refresh rate for smoother animation
            expand=True
        )
        
        # Spinner animations for active workers
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        
        # Main progress task
        self.overall_task = self.progress.add_task(
            "[green]Processing videos", 
            total=100, 
            spinner="",
            status="Starting..."
        )
        
        # Worker tasks
        self.worker_tasks = {}
        for i in range(num_workers):
            task_id = self.progress.add_task(
                f"Worker {i+1}", 
                total=100,  # Keep total as 100 
                completed=0,
                spinner="",  # Empty spinner when idle
                status="Idle"
            )
            self.worker_tasks[i] = task_id
        
        # Start the progress display
        self.progress.start()
    
    def set_total_videos(self, total):
        """Set the total number of videos to process"""
        self.total_videos = total
        self.progress.update(
            self.overall_task, 
            total=total, 
            status=f"0/{total} videos"
        )
    
    def update_worker_status(self, worker_id, status, video_index, percent_done=None):
        """Update status for a specific worker"""
        # Check if we should actually update the display
        current_time = time.time()
        should_refresh = (current_time - self.last_refresh) >= self.refresh_rate
        
        if worker_id < len(self.worker_status):
            # Store the previous status to check for significant changes
            previous_status = self.worker_status[worker_id]
            
            # Store the new status
            self.worker_status[worker_id] = status
            
            # Track video progress
            if percent_done is not None:
                self.video_progress[video_index] = percent_done
            
            # Apply appropriate styling to status text
            styled_status = status
            is_active = False  # Track if worker is active
            
            # Don't cut status message so much - allow longer messages
            if len(styled_status) > 95:
                styled_status = styled_status[:92] + "..."
            
            # Detect significant status changes that should force an update
            # This happens when a worker changes from Idle to Starting or stops being Idle
            force_update = False
            if "Idle" in previous_status and "Starting" in status:
                force_update = True
            elif "Starting" in previous_status:
                force_update = True
            
            # Special handling for completion events (always update these immediately)
            immediate_update = False
            
            # Update spinner based on status
            spinner_char = ""
            
            if "MOTION DETECTED" in status:
                styled_status = f"[green]{styled_status}[/green]"
                self.motion_found += 1
                self.completed += 1
                spinner_char = "✓"  # Checkmark for completed with motion
                immediate_update = True
                
                # Update overall progress
                self.progress.update(
                    self.overall_task, 
                    completed=self.completed, 
                    status=f"{self.completed}/{self.total_videos} videos • {self.motion_found} with motion"
                )
                
            elif "Error" in status:
                styled_status = f"[red]{styled_status}[/red]"
                spinner_char = "✘"  # X mark for error
                immediate_update = True
                
            elif "No motion" in status:
                styled_status = f"[yellow]{styled_status}[/yellow]"
                self.completed += 1
                spinner_char = "○"  # Empty circle for no motion
                immediate_update = True
                
                # Update overall progress
                self.progress.update(
                    self.overall_task, 
                    completed=self.completed, 
                    status=f"{self.completed}/{self.total_videos} videos • {self.motion_found} with motion"
                )
                
            elif "Processing" in status:
                styled_status = f"[blue]{styled_status}[/blue]"
                # Use a rotating spinner frame for active processing
                spinner_char = self.spinner_frames[self.spinner_index % len(self.spinner_frames)]
                is_active = True
                force_update = True  # Always update on processing status
                
            elif "Aborted" in status:
                styled_status = f"[yellow]{styled_status}[/yellow]"
                spinner_char = "⨯"  # X mark for aborted
                immediate_update = True
            
            # Only update the display if enough time has passed or it's a completion event or forced update
            if should_refresh or immediate_update or force_update:
                # Update worker task with spinner and status
                task_id = self.worker_tasks[worker_id]
                self.progress.update(
                    task_id,
                    spinner=spinner_char,
                    status=styled_status
                )
                
                # Force the progress display to refresh
                self.progress.refresh()
                
                # Update the refresh timestamp
                self.last_refresh = current_time
            
            # Advance spinner animation frame after each update for active workers
            if is_active:
                self.spinner_index += 1

    def final_summary(self, motion_videos):
        """Display final summary after processing is complete"""
        # Stop the progress display
        self.progress.stop()
        
        self.console.print("\n[bold green]Processing Complete![/bold green]")
        self.console.print(f"Total videos analyzed: {self.total_videos}")
        self.console.print(f"Videos with motion detected: {len(motion_videos)}")
        self.console.print(f"Total time: {format_time(time.time() - self.start_time)}")
        
        if motion_videos:
            self.console.print("\n[bold]Videos with substantial motion:[/bold]")
            for i, video in enumerate(motion_videos):
                self.console.print(f"{i+1}. {video}")

def extract_timestamp_for_sorting(filename):
    """Extract timestamp from filename for sorting purposes, returns empty string if no match"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', basename)
    if match:
        return match.group(1)
    return ""

def extract_camera_type(filename):
    """Extract camera type from the filename based on the pattern."""
    # Remove path and extension
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    
    # Tesla Sentry Mode format: YYYY-MM-DD_HH-MM-SS-camera_type.mp4
    # Example: 2025-04-05_10-57-50-front.mp4
    
    # Try to match the date-time pattern first
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(.+)'
    match = re.match(pattern, name)
    
    if match:
        camera_type = match.group(1)
        return camera_type
    
    # If pattern doesn't match, return "unknown"
    return "unknown"

def should_process_camera(filename, filter_cameras=None):
    """Check if a file should be processed based on camera filter."""
    if not filter_cameras:
        return True
        
    camera_type = extract_camera_type(filename)
    return camera_type in filter_cameras

def extract_timestamp(filename):
    """Extract timestamp from the filename."""
    # Remove path and extension
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    
    # Try to match the pattern: YYYY-MM-DD_HH-MM-SS-camera_type.mp4
    match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', name)
    if match:
        return match.group(1)
    
    return ""

def extract_thumbnails(video_path, num_thumbnails=4):
    """
    Extract thumbnails from a video.
    
    Args:
        video_path: Path to the video file
        num_thumbnails: Number of thumbnails to extract
    
    Returns:
        List of thumbnail images as base64 encoded strings or None if failed
    """
    try:
        thumbnails = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or duration <= 0:
            cap.release()
            return None
        
        # Calculate frames to capture
        frames_to_capture = []
        for i in range(num_thumbnails):
            # Distribute evenly across the video duration
            # Skip the first and last few frames
            pos = max(0.1, min(0.9, (i + 1) / (num_thumbnails + 1)))
            frame_pos = int(frame_count * pos)
            frames_to_capture.append(frame_pos)
        
        # Extract the thumbnails
        for frame_pos in frames_to_capture:
            # Set position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Resize to a reasonable thumbnail size
            height, width = frame.shape[:2]
            max_dimension = 320  # Max width or height
            
            # Calculate scale factor to maintain aspect ratio
            scale = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize frame
            thumbnail = cv2.resize(frame, (new_width, new_height))
            
            # Convert to base64 for HTML embedding
            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            thumbnails.append(jpg_as_text)
        
        cap.release()
        return thumbnails
        
    except Exception as e:
        print(f"Error extracting thumbnails from {video_path}: {e}")
        return None

def generate_html_report(motion_videos, output_file, group_by='both', root_path=None):
    """
    Generate an HTML report with the motion videos, sorted by folder and camera type.
    
    Args:
        motion_videos: List of paths to videos with motion
        output_file: Path to save the HTML report
        group_by: How to group results: by folder, by camera type, or both (default: both)
        root_path: Root path for creating relative links
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Group videos by folder and/or camera type based on the group_by parameter
        videos_by_folder = defaultdict(lambda: defaultdict(list))
        videos_by_camera = defaultdict(list)
        
        for video_path in motion_videos:
            folder = os.path.dirname(video_path)
            camera_type = extract_camera_type(video_path)
            
            # Store in appropriate structures based on grouping
            if group_by in ['both', 'folder']:
                videos_by_folder[folder][camera_type].append(video_path)
            
            if group_by in ['both', 'camera']:
                videos_by_camera[camera_type].append(video_path)
        
        # Sort folders and camera types
        sorted_folders = sorted(videos_by_folder.keys())
        sorted_cameras = sorted(videos_by_camera.keys())
        
        # Sort videos by timestamp within each group (newest first)
        for folder in sorted_folders:
            for camera_type in videos_by_folder[folder]:
                videos_by_folder[folder][camera_type].sort(key=extract_timestamp_for_sorting, reverse=True)
        
        for camera_type in sorted_cameras:
            videos_by_camera[camera_type].sort(key=extract_timestamp_for_sorting, reverse=True)
        
        # Start building HTML
        html = []
        html.append("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tesla Sentry Mode Motion Detection Results</title>
            <script>
                // Set up a JavaScript refresh timer instead of using meta refresh
                var refreshTimer = setTimeout(function() {
                    window.location.reload();
                }, 5000); // 5 seconds refresh
                var autoRefreshEnabled = true;
                
                // Function to toggle auto-refresh
                function stopAutoRefresh() {
                    if (autoRefreshEnabled) {
                        // Stop auto-refresh
                        clearTimeout(refreshTimer);
                        autoRefreshEnabled = false;
                        
                        // Update the UI
                        document.getElementById('refresh-status').innerText = 'Auto-refresh disabled';
                        document.getElementById('stop-refresh-btn').innerText = 'Start Auto-Refresh';
                        document.getElementById('stop-refresh-btn').style.backgroundColor = '#4caf50'; // Green for Start
                        
                        // Save the preference in localStorage
                        try {
                            localStorage.setItem('autoRefreshDisabled', 'true');
                        } catch (e) {
                            // Local storage might not be available, proceed without it
                        }
                    } else {
                        // Start auto-refresh
                        refreshTimer = setTimeout(function() {
                            window.location.reload();
                        }, 5000); // 5 seconds
                        autoRefreshEnabled = true;
                        
                        // Update the UI
                        document.getElementById('refresh-status').innerText = 'Page refreshes every 5 seconds';
                        document.getElementById('stop-refresh-btn').innerText = 'Stop Auto-Refresh';
                        document.getElementById('stop-refresh-btn').style.backgroundColor = '#ff6b6b'; // Red for Stop
                        
                        // Save the preference in localStorage
                        try {
                            localStorage.setItem('autoRefreshDisabled', 'false');
                        } catch (e) {
                            // Local storage might not be available, proceed without it
                        }
                    }
                    
                    return false; // Prevent form submission
                }
                
                // Check for previously saved preference
                document.addEventListener('DOMContentLoaded', function() {
                    try {
                        if (localStorage.getItem('autoRefreshDisabled') === 'true') {
                            // Auto-refresh was previously disabled
                            clearTimeout(refreshTimer);
                            autoRefreshEnabled = false;
                            document.getElementById('refresh-status').innerText = 'Auto-refresh disabled';
                            document.getElementById('stop-refresh-btn').innerText = 'Start Auto-Refresh';
                            document.getElementById('stop-refresh-btn').style.backgroundColor = '#4caf50'; // Green for Start
                        }
                    } catch (e) {
                        // Local storage might not be available, proceed without it
                    }
                    
                    // Simple JavaScript for filtering
                    const filterButtons = document.querySelectorAll('.filter-btn');
                    
                    filterButtons.forEach(btn => {
                        btn.addEventListener('click', function() {
                            const filter = this.getAttribute('data-filter');
                            
                            // Toggle active class
                            filterButtons.forEach(b => b.classList.remove('active'));
                            this.classList.add('active');
                            
                            // Hide/show sections based on filter
                            if (filter === 'all') {
                                document.querySelectorAll('.folder-section, .camera-section').forEach(
                                    section => section.style.display = 'block'
                                );
                            } else if (filter === 'folders') {
                                document.querySelectorAll('.folder-section').forEach(
                                    section => section.style.display = 'block'
                                );
                                document.querySelectorAll('.camera-section').forEach(
                                    section => section.style.display = 'none'
                                );
                            } else if (filter === 'cameras') {
                                document.querySelectorAll('.folder-section').forEach(
                                    section => section.style.display = 'none'
                                );
                                document.querySelectorAll('.camera-section').forEach(
                                    section => section.style.display = 'block'
                                );
                            }
                        });
                    });
                });
            </script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }
                h2 {
                    color: #555;
                    margin-top: 30px;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                }
                h3 {
                    color: #777;
                    margin-top: 20px;
                }
                .video-container {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 15px;
                }
                .thumbnails {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 10px;
                    flex-wrap: wrap;
                }
                .thumbnail {
                    margin: 5px;
                    border: 1px solid #ddd;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .timestamp {
                    color: #888;
                    font-size: 0.9em;
                }
                .folder-path {
                    font-family: monospace;
                    background-color: #eee;
                    padding: 5px;
                    border-radius: 3px;
                    font-size: 0.9em;
                    overflow-wrap: break-word;
                }
                .video-link {
                    display: block;
                    margin-top: 5px;
                    font-family: monospace;
                    color: #0066cc;
                    text-decoration: none;
                }
                .video-link:hover {
                    text-decoration: underline;
                }
                .summary {
                    background-color: #e9f7fe;
                    border-left: 4px solid #1e88e5;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 3px;
                }
                .filters {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 20px 0;
                }
                .filter-btn {
                    padding: 8px 16px;
                    background-color: #f1f1f1;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }
                .filter-btn:hover {
                    background-color: #e1e1e1;
                }
                .filter-btn.active {
                    background-color: #1e88e5;
                    color: white;
                }
                #toc {
                    background-color: #fff;
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }
                .toc-title {
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .toc-list {
                    list-style-type: none;
                    padding-left: 10px;
                }
                .toc-list li {
                    margin-bottom: 5px;
                }
                .toc-list a {
                    text-decoration: none;
                    color: #0066cc;
                }
                .toc-list a:hover {
                    text-decoration: underline;
                }
                .control-button {
                    padding: 10px 15px;
                    background-color: #ff6b6b;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-right: 10px;
                    font-weight: bold;
                }
                .control-button:hover {
                    background-color: #ff5252;
                }
                #controls {
                    margin-bottom: 20px;
                }
                .note {
                    background-color: #e8f5e9;
                    border-left: 4px solid #4caf50;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 3px;
                    font-style: italic;
                }
            </style>
        </head>
        <body>
            <h1>Tesla Sentry Mode Motion Detection Results</h1>
            
            <div id="controls">
                <button id="stop-refresh-btn" class="control-button" onclick="stopAutoRefresh()">Stop Auto-Refresh</button>
                <span id="refresh-status">Page refreshes every 5 seconds</span>
            </div>
            
            <div class="note">
                This report uses relative paths and should be kept in this folder to maintain working video links.
            </div>
            
            <div class="summary">
                <p>Total folders: <b>""" + str(len(sorted_folders)) + """</b></p>
                <p>Total cameras: <b>""" + str(len(sorted_cameras)) + """</b></p>
                <p>Total videos with motion: <b>""" + str(len(motion_videos)) + """</b></p>
                <p>Report generated: <b>""" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</b></p>
            </div>
        """)
        
        # Add filters if both grouping methods are available
        if group_by == 'both':
            html.append("""
            <div class="filters">
                <button class="filter-btn active" data-filter="all">Show All</button>
                <button class="filter-btn" data-filter="folders">Group by Folder</button>
                <button class="filter-btn" data-filter="cameras">Group by Camera</button>
            </div>
            """)
        
        # Add table of contents
        html.append('<div id="toc">')
        html.append('<div class="toc-title">Table of Contents</div>')
        html.append('<ul class="toc-list">')
        
        # Add folder links to TOC if grouping by folder
        if group_by in ['both', 'folder'] and sorted_folders:
            html.append('<li><strong>Folders</strong></li>')
            for folder in sorted_folders:
                folder_id = 'folder-' + str(hash(folder) % 10000000)
                html.append(f'<li><a href="#{folder_id}">{os.path.basename(folder) or folder}</a></li>')
        
        # Add camera links to TOC if grouping by camera
        if group_by in ['both', 'camera'] and sorted_cameras:
            html.append('<li><strong>Cameras</strong></li>')
            for camera in sorted_cameras:
                camera_id = 'camera-' + camera.replace('_', '-')
                html.append(f'<li><a href="#{camera_id}">{camera}</a></li>')
        
        html.append('</ul>')
        html.append('</div>')
        
        # Process by folder if requested
        if group_by in ['both', 'folder']:
            for folder in sorted_folders:
                folder_id = 'folder-' + str(hash(folder) % 10000000)
                html.append(f"""
                <div class="folder-section">
                    <h2 id="{folder_id}">Folder</h2>
                    <div class="folder-path">{folder}</div>
                """)
                
                # Sort camera types
                camera_types = sorted(videos_by_folder[folder].keys())
                
                for camera_type in camera_types:
                    videos = videos_by_folder[folder][camera_type]
                    
                    # Sort videos by timestamp
                    videos.sort()
                    
                    html.append(f"""
                    <h3>Camera: {camera_type} ({len(videos)} videos)</h3>
                    """)
                    
                    # Process each video in this folder and camera type
                    for video_path in videos:
                        html.append(create_video_html(video_path, root_path))
                
                html.append('</div>') # Close folder-section
        
        # Process by camera if requested
        if group_by in ['both', 'camera']:
            for camera_type in sorted_cameras:
                camera_id = 'camera-' + camera_type.replace('_', '-')
                html.append(f"""
                <div class="camera-section">
                    <h2 id="{camera_id}">Camera: {camera_type}</h2>
                """)
                
                videos = videos_by_camera[camera_type]
                
                # Sort videos by folder and then by name
                videos.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))
                
                # Group by folder for better organization
                folder_to_videos = defaultdict(list)
                for video in videos:
                    folder_to_videos[os.path.dirname(video)].append(video)
                
                # Display videos grouped by folder
                for folder, folder_videos in sorted(folder_to_videos.items()):
                    html.append(f"""
                    <h3>Folder: {os.path.basename(folder) or folder}</h3>
                    <div class="folder-path">{folder}</div>
                    """)
                    
                    for video_path in sorted(folder_videos):
                        html.append(create_video_html(video_path, root_path))
                
                html.append('</div>') # Close camera-section
        
        # Close HTML
        html.append("""
        </body>
        </html>
        """)
        
        # Write the HTML to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(html))
        
        return True
    
    except Exception as e:
        if USE_RICH:
            Console().print(f"[{get_timestamp()}] Error generating HTML report: {e}", style="red")
        else:
            print(f"[{get_timestamp()}] Error generating HTML report: {e}")
        return False

def create_video_html(video_path, root_path=None):
    """Create HTML for a single video entry"""
    # Extract video details
    basename = os.path.basename(video_path)
    timestamp = extract_timestamp(video_path)
    
    # Create relative path if root_path is provided
    if root_path and os.path.isabs(video_path) and video_path.startswith(root_path):
        # Calculate relative path from root_path
        rel_path = os.path.relpath(video_path, root_path)
        # Use relative link
        link_path = rel_path
    else:
        # Use absolute path with file:// protocol
        link_path = f"file://{video_path}"
    
    # Build HTML
    video_html = f"""
    <div class="video-container">
        <div class="timestamp">{timestamp}</div>
        <a href="{link_path}" class="video-link">{basename}</a>
    """
    
    # Add thumbnails if they exist
    thumbnails = extract_thumbnails(video_path)
    if thumbnails and len(thumbnails) > 0:
        video_html += '<div class="thumbnails">'
        for thumb in thumbnails:
            video_html += f'<img src="data:image/jpeg;base64,{thumb}" class="thumbnail">'
        video_html += '</div>'
    
    video_html += '</div>'
    
    return video_html

class HTMLProgressWriter:
    """Class to handle real-time updates to the HTML report"""
    
    def __init__(self, output_file, group_by, root_path):
        self.output_file = output_file
        self.group_by = group_by
        self.root_path = root_path
        self.motion_videos = []
        self.videos_by_folder = defaultdict(lambda: defaultdict(list))
        self.videos_by_camera = defaultdict(list)
        self.last_update_time = time.time()
        self.update_interval = 30  # Update every 30 seconds, changed back from 5 seconds
        self.total_processed = 0
        self.total_videos = 0
    
    def set_progress_counts(self, processed, total):
        """Set the number of processed videos and total videos."""
        self.total_processed = processed
        self.total_videos = total
        logger.info(f"Set HTML report progress counts: {processed}/{total} videos")
    
    def add_video(self, video_path):
        """Add a video with motion to the report."""
        # Add to our list of videos
        self.motion_videos.append(video_path)
        
        # Group by folder and camera
        folder = os.path.dirname(video_path)
        camera_type = extract_camera_type(video_path)
        
        # Store in grouping structures
        if self.group_by in ['both', 'folder']:
            self.videos_by_folder[folder][camera_type].append(video_path)
        
        if self.group_by in ['both', 'camera']:
            self.videos_by_camera[camera_type].append(video_path)
        
        # Always update HTML immediately when a new video with motion is detected
        self.update_html()
        logger.info(f"HTML report updated with new motion video: {video_path}")
        self.last_update_time = time.time()
        
        # We'll also keep the periodic updates in case multiple videos are detected in quick succession
        # this ensures we don't update too frequently and slow down processing
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self.update_html()
            logger.info(f"Periodic HTML update performed. Total videos: {len(self.motion_videos)}")
            self.last_update_time = current_time
    
    def update_html(self):
        """Update the HTML file with current progress."""
        try:
            # Sort folders and camera types
            sorted_folders = sorted(self.videos_by_folder.keys())
            sorted_cameras = sorted(self.videos_by_camera.keys())
            
            # Sort videos within each group by timestamp (newest first)
            for folder in sorted_folders:
                for camera_type in self.videos_by_folder[folder]:
                    self.videos_by_folder[folder][camera_type].sort(key=extract_timestamp_for_sorting, reverse=True)
            
            for camera_type in sorted_cameras:
                self.videos_by_camera[camera_type].sort(key=extract_timestamp_for_sorting, reverse=True)
            
            # Get current time for last updated timestamp
            last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get the progress tracking variables
            progress_info = {}
            try:
                # First try to get from main progress tracker
                from __main__ import progress
                progress_info['total_processed'] = progress.completed if hasattr(progress, 'completed') else 0
                progress_info['total_videos'] = progress.total_videos if hasattr(progress, 'total_videos') else 0
            except:
                # Fall back to our own progress tracking
                progress_info['total_processed'] = self.total_processed
                progress_info['total_videos'] = self.total_videos
            
            # Get the common HTML components
            components = generate_html_components(
                self.motion_videos, 
                sorted_folders, 
                sorted_cameras,
                self.group_by,
                f"In Progress ({progress_info['total_processed']}/{progress_info['total_videos']} videos)" if progress_info.get('total_videos', 0) > 0 else "In Progress",
                last_updated,
                self.update_interval
            )
            
            # Start building the complete HTML
            html = [components['head'], components['summary'], components['filters'], components['toc']]
            
            # Process by folder if requested
            if self.group_by in ['both', 'folder']:
                for folder in sorted_folders:
                    folder_id = 'folder-' + str(hash(folder) % 10000000)
                    html.append(f"""
                    <div class="folder-section">
                        <h2 id="{folder_id}">Folder</h2>
                        <div class="folder-path">{folder}</div>
                    """)
                    
                    # Sort camera types
                    camera_types = sorted(self.videos_by_folder[folder].keys())
                    
                    for camera_type in camera_types:
                        videos = self.videos_by_folder[folder][camera_type]
                        
                        # Sort videos by timestamp
                        videos.sort()
                        
                        html.append(f"""
                        <h3>Camera: {camera_type} ({len(videos)} videos)</h3>
                        """)
                        
                        # Process each video in this folder and camera type
                        for video_path in videos:
                            html.append(create_video_html(video_path, self.root_path))
                    
                    html.append('</div>') # Close folder-section
            
            # Process by camera if requested
            if self.group_by in ['both', 'camera']:
                for camera_type in sorted_cameras:
                    camera_id = 'camera-' + camera_type.replace('_', '-')
                    html.append(f"""
                    <div class="camera-section">
                        <h2 id="{camera_id}">Camera: {camera_type}</h2>
                    """)
                    
                    videos = self.videos_by_camera[camera_type]
                    
                    # Sort videos by folder and then by name
                    videos.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))
                    
                    # Group by folder for better organization
                    folder_to_videos = defaultdict(list)
                    for video in videos:
                        folder_to_videos[os.path.dirname(video)].append(video)
                    
                    # Display videos grouped by folder
                    for folder, folder_videos in sorted(folder_to_videos.items()):
                        html.append(f"""
                        <h3>Folder: {os.path.basename(folder) or folder}</h3>
                        <div class="folder-path">{folder}</div>
                        """)
                        
                        for video_path in sorted(folder_videos):
                            html.append(create_video_html(video_path, self.root_path))
                    
                    html.append('</div>') # Close camera-section
            
            # Close HTML
            html.append(components['close'])
            
            # Write the updated HTML file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(html))
                
            return True
        
        except Exception as e:
            log_error("Error updating HTML report", e)
            return False
    
    def finalize(self):
        """Finalize the HTML report and remove the progress indicator."""
        try:
            # Do a final update
            self.update_html()
            logger.info(f"Finalizing HTML report with {len(self.motion_videos)} videos")
            
            # Read the current HTML file
            with open(self.output_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # No need to disable auto-refresh - let user control it with the toggle button
            
            # Remove the progress info div if it exists
            html_content = re.sub(
                r'<div class="progress-info">.*?</div>',
                '',
                html_content,
                flags=re.DOTALL
            )
            
            # Update the status to completed (including progress counts if present)
            status_pattern = r'<p>Status: <b>In Progress(?: \(\d+/\d+ videos\))?</b></p>'
            completed_status = f'<p>Status: <b>Completed ({self.total_processed}/{self.total_videos} videos)</b></p>' if self.total_videos > 0 else '<p>Status: <b>Completed</b></p>'
            html_content = re.sub(status_pattern, completed_status, html_content)
            
            # Update the summary to indicate completion
            html_content = html_content.replace(
                'Total videos with motion found so far:',
                'Total videos with motion:'
            )
            
            # Add completion time
            completion_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_content = html_content.replace(
                'Report generated:',
                f'Report completed: <b>{completion_time}</b><br>Report generated:'
            )
            
            # Write the updated HTML file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report finalized successfully with {len(self.motion_videos)} videos")
            return True
            
        except Exception as e:
            log_error("Error finalizing HTML report", e)
            return False

def find_motion_videos(root_path, extension='.mp4', threshold=45, min_area_percentage=2.5, 
                       max_workers=8, sample_frames=True, use_gpu=True, html_writer=None, already_processed=set(), limit=0, filter_cameras=None):
    """
    Find all videos with substantial motion under the root path.
    
    Args:
        root_path: Root directory to search for videos
        extension: File extension to look for
        threshold: Threshold for motion detection
        min_area_percentage: Minimum percentage of frame that needs to show motion
        max_workers: Maximum number of parallel workers
        sample_frames: If True, sample frames instead of processing all frames
        use_gpu: If True, use GPU acceleration if available
        html_writer: HTMLProgressWriter instance to update HTML in real-time
        already_processed: Set of video paths that have already been processed
        limit: Limit the number of videos to process (0 means no limit)
        filter_cameras: List of camera types to process (None for all cameras)
    
    Returns:
        List of paths to videos with substantial motion
    """
    global abort_requested
    motion_videos = []
    video_paths = []
    futures = []
    executor = None
    skipped_camera_count = 0
    
    try:
        if USE_RICH:
            console = Console()
            console.print(f"[{get_timestamp()}] Starting motion detection process")
            
            # Check GPU availability
            cuda_available = has_cuda_support()
            if use_gpu:
                if cuda_available:
                    console.print(f"[{get_timestamp()}] CUDA GPU acceleration enabled", style="green")
                else:
                    console.print(f"[{get_timestamp()}] GPU acceleration requested but no CUDA support found. Falling back to CPU.", style="yellow")
            
            # Show camera filter information
            if filter_cameras:
                console.print(f"[{get_timestamp()}] Filtering for camera types: {', '.join(filter_cameras)}", style="green")
            
            # Find all video files with the specified extension
            console.print(f"[{get_timestamp()}] Scanning for {extension} files in {root_path}")
            scan_start = time.time()
            
            for dirpath, _, filenames in os.walk(root_path):
                for filename in filenames:
                    if filename.lower().endswith(extension.lower()):
                        file_path = os.path.join(dirpath, filename)
                        # Apply camera filter before adding to processing list
                        if should_process_camera(file_path, filter_cameras):
                            video_paths.append(file_path)
                        else:
                            skipped_camera_count += 1
            
            scan_elapsed = time.time() - scan_start
            
            if not video_paths:
                if skipped_camera_count > 0:
                    console.print(f"[{get_timestamp()}] No matching {extension} files found for the specified camera types. Skipped {skipped_camera_count} videos from other cameras.", style="yellow")
                else:
                    console.print(f"[{get_timestamp()}] No {extension} files found in {root_path}", style="red")
                return motion_videos
            
            # Report on skipped cameras
            if skipped_camera_count > 0:
                console.print(f"[{get_timestamp()}] Skipped {skipped_camera_count} videos that didn't match the camera filter", style="yellow")
            
            # Sort videos from newest to oldest based on filename timestamp
            video_paths.sort(key=extract_timestamp_for_sorting, reverse=True)
            
            console.print(f"[{get_timestamp()}] Found {len(video_paths)} matching video files in {scan_elapsed:.1f}s. Starting motion detection...")
            console.print(f"[{get_timestamp()}] Using {max_workers} parallel workers\n")
            console.print(f"[{get_timestamp()}] You can press Ctrl+C or create an '{abort_file}' file to gracefully abort processing.", style="yellow")
            
            # Initialize progress tracker
            progress = RichProgressTracker(max_workers)
        else:
            # Using simple console output
            print(f"[{get_timestamp()}] Starting motion detection process")
            
            # Check GPU availability
            cuda_available = has_cuda_support()
            if use_gpu:
                if cuda_available:
                    print(f"[{get_timestamp()}] CUDA GPU acceleration enabled")
                else:
                    print(f"[{get_timestamp()}] GPU acceleration requested but no CUDA support found. Falling back to CPU.")
            
            # Show camera filter information
            if filter_cameras:
                print(f"[{get_timestamp()}] Filtering for camera types: {', '.join(filter_cameras)}")
            
            # Find all video files with the specified extension
            print(f"[{get_timestamp()}] Scanning for {extension} files in {root_path}")
            scan_start = time.time()
            
            for dirpath, _, filenames in os.walk(root_path):
                for filename in filenames:
                    if filename.lower().endswith(extension.lower()):
                        file_path = os.path.join(dirpath, filename)
                        # Apply camera filter before adding to processing list
                        if should_process_camera(file_path, filter_cameras):
                            video_paths.append(file_path)
                        else:
                            skipped_camera_count += 1
            
            scan_elapsed = time.time() - scan_start
            
            if not video_paths:
                if skipped_camera_count > 0:
                    print(f"[{get_timestamp()}] No matching {extension} files found for the specified camera types. Skipped {skipped_camera_count} videos from other cameras.")
                else:
                    print(f"[{get_timestamp()}] No {extension} files found in {root_path}")
                return motion_videos
            
            # Report on skipped cameras
            if skipped_camera_count > 0:
                print(f"[{get_timestamp()}] Skipped {skipped_camera_count} videos that didn't match the camera filter")
            
            # Sort videos from newest to oldest based on filename timestamp
            video_paths.sort(key=extract_timestamp_for_sorting, reverse=True)
            
            print(f"[{get_timestamp()}] Found {len(video_paths)} matching video files in {scan_elapsed:.1f}s. Starting motion detection...")
            print(f"[{get_timestamp()}] Using {max_workers} parallel workers\n")
            print(f"[{get_timestamp()}] You can press Ctrl+C or create an '{abort_file}' file to gracefully abort processing.")
            
            # Initialize progress tracker
            progress = SimpleProgressTracker(max_workers)
            
        # Set total videos to process
        progress.set_total_videos(len(video_paths))
        
        # If we have an HTML writer, update it with the total number of videos
        if html_writer:
            html_writer.set_progress_counts(0, len(video_paths))
        
        # Define progress updater function
        def update_progress(worker_id, status, video_index, percent_done=None):
            progress.update_worker_status(worker_id, status, video_index, percent_done)
            
            # If we have an HTML writer, update it with the latest progress counts
            if html_writer and hasattr(progress, 'completed'):
                html_writer.set_progress_counts(progress.completed, progress.total_videos)
        
        # Start the abort file checker thread
        abort_thread = threading.Thread(target=check_for_abort_file, daemon=True)
        abort_thread.start()
        
        # Process videos in parallel
        executor = ThreadPoolExecutor(max_workers=max_workers)
        # Map each video path to a future with index and worker information
        future_to_path = {}
        
        for idx, path in enumerate(video_paths):
            # Check for abort request
            if abort_requested:
                break
                
            # Find an available worker ID
            worker_id = idx % max_workers
            
            # Skip if already processed
            if path in already_processed:
                continue
            
            # Submit task
            future = executor.submit(
                detect_motion, 
                path, 
                threshold, 
                min_area_percentage, 
                sample_frames,
                use_gpu,
                idx,
                len(video_paths),
                worker_id,
                update_progress,
                html_writer
            )
            
            future_to_path[future] = (path, idx, worker_id)
            futures.append(future)
            
            # Slight delay between submissions to spread out initial work
            if idx < max_workers:
                time.sleep(0.1)
            
            # Check if we've reached the limit
            if limit > 0 and len(motion_videos) >= limit:
                break
        
        # Process the results as they complete
        for future in as_completed(futures):
            # Check for abort request
            if abort_requested:
                break
                
            path, idx, worker_id = future_to_path[future]
            
            try:
                has_motion = future.result()
                if has_motion:
                    motion_videos.append(path)
            except Exception as e:
                if USE_RICH:
                    console.print(f"[{get_timestamp()}] Error processing {path}: {e}", style="red")
                else:
                    print(f"[{get_timestamp()}] Error processing {path}: {e}")
        
        # Update HTML writer with final counts before displaying summary
        if html_writer and hasattr(progress, 'completed'):
            html_writer.set_progress_counts(progress.completed, progress.total_videos)
            
        # Display final summary if not aborted
        if not abort_requested:
            progress.final_summary(motion_videos)
        
    except KeyboardInterrupt:
        # Already handled by signal handler
        pass
    
    finally:
        # Cancel any pending futures and shutdown the executor
        if executor:
            if abort_requested:
                # Cancel all pending tasks
                for future in futures:
                    if not future.done():
                        future.cancel()
                
                if USE_RICH:
                    console.print(f"[{get_timestamp()}] Processing aborted. Shutting down workers...", style="yellow")
                else:
                    print(f"[{get_timestamp()}] Processing aborted. Shutting down workers...")
                    
                # Shutdown the executor without waiting
                executor.shutdown(wait=False)
            else:
                # Normal shutdown
                executor.shutdown(wait=True)
    
    return motion_videos

def format_time(seconds):
    """Format seconds into a human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def print_configuration(args):
    """Print the configuration being used."""
    if USE_RICH:
        console = Console()
        console.print(f"[{get_timestamp()}] Using configuration:")
        console.print(f"  - Root path: {args.root_path}")
        console.print(f"  - File extension: {args.extension}")
        console.print(f"  - Motion threshold: {args.threshold}")
        console.print(f"  - Minimum area percentage: {args.min_area}%")
        console.print(f"  - Parallel workers: {args.workers}")
        console.print(f"  - Output file: {args.output}")
        console.print(f"  - Sample frames: {'Yes' if args.sample_frames else 'No'}")
        console.print(f"  - GPU acceleration: {'Yes (if available)' if args.use_gpu else 'No'}")
        console.print(f"  - Generate thumbnails: {'Yes' if hasattr(args, 'generate_thumbnails') and args.generate_thumbnails else 'No'}")
        console.print(f"  - Filter by camera: {args.filter_camera if hasattr(args, 'filter_camera') and args.filter_camera else 'None'}")
        console.print(f"  - Group by: {args.group_by if hasattr(args, 'group_by') else 'both'}")
        console.print(f"  - Real-time HTML updates: {'Yes' if hasattr(args, 'realtime_updates') and args.realtime_updates else 'No'}")
        console.print()
    else:
        print(f"[{get_timestamp()}] Using configuration:")
        print(f"  - Root path: {args.root_path}")
        print(f"  - File extension: {args.extension}")
        print(f"  - Motion threshold: {args.threshold}")
        print(f"  - Minimum area percentage: {args.min_area}%")
        print(f"  - Parallel workers: {args.workers}")
        print(f"  - Output file: {args.output}")
        print(f"  - Sample frames: {'Yes' if args.sample_frames else 'No'}")
        print(f"  - GPU acceleration: {'Yes (if available)' if args.use_gpu else 'No'}")
        print(f"  - Generate thumbnails: {'Yes' if hasattr(args, 'generate_thumbnails') and args.generate_thumbnails else 'No'}")
        print(f"  - Filter by camera: {args.filter_camera if hasattr(args, 'filter_camera') and args.filter_camera else 'None'}")
        print(f"  - Group by: {args.group_by if hasattr(args, 'group_by') else 'both'}")
        print(f"  - Real-time HTML updates: {'Yes' if hasattr(args, 'realtime_updates') and args.realtime_updates else 'No'}")
        print()

def write_results_file(filename, motion_videos):
    """Simple function to write results to a file"""
    try:
        print(f"Attempting to write to: {filename}")
        with open(filename, "w") as f:
            for video in motion_videos:
                f.write(f"{video}\n")
        print(f"Successfully wrote results to: {filename}")
        return True
    except Exception as e:
        print(f"Failed to write results: {e}")
        return False

def log_error(message, exception=None, style="red"):
    """
    Log an error message consistently to both console and log file.
    
    Args:
        message: The error message to display
        exception: Optional exception object to include details from
        style: The rich console style to use (if rich is available)
    """
    error_text = message
    if exception:
        error_text = f"{message}: {str(exception)}"
        
    logger.error(error_text)
    
    if USE_RICH:
        Console().print(f"[{get_timestamp()}] {error_text}", style=style)
    else:
        print(f"[{get_timestamp()}] {error_text}")

def generate_html_components(motion_videos, sorted_folders, sorted_cameras, group_by='both', 
                           status="Completed", last_updated=None, auto_refresh_interval=30):
    """
    Generate common HTML components used in both the initial and final HTML reports.
    
    Args:
        motion_videos: List of paths to videos with motion
        sorted_folders: Sorted list of folder paths
        sorted_cameras: Sorted list of camera types
        group_by: How to group results: by folder, by camera type, or both (default: both)
        status: Current status (In Progress or Completed)
        last_updated: When the report was last updated (datetime string)
        auto_refresh_interval: Auto-refresh interval in seconds
        
    Returns:
        Dictionary with HTML components (head, summary, filters, toc, folders, cameras)
    """
    if not last_updated:
        last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    generation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build HTML components
    components = {}
    
    # HTML head section with styles and scripts
    components['head'] = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tesla Sentry Mode Motion Detection Results</title>
        <script>
            // Set up a JavaScript refresh timer instead of using meta refresh
            var refreshTimer = setTimeout(function() {{
                window.location.reload();
            }}, {auto_refresh_interval * 1000}); // {auto_refresh_interval} seconds refresh
            var autoRefreshEnabled = true;
            
            // Function to toggle auto-refresh
            function stopAutoRefresh() {{
                if (autoRefreshEnabled) {{
                    // Stop auto-refresh
                    clearTimeout(refreshTimer);
                    autoRefreshEnabled = false;
                    
                    // Update the UI
                    document.getElementById('refresh-status').innerText = 'Auto-refresh disabled';
                    document.getElementById('stop-refresh-btn').innerText = 'Start Auto-Refresh';
                    document.getElementById('stop-refresh-btn').style.backgroundColor = '#4caf50'; // Green for Start
                    
                    // Save the preference in localStorage
                    try {{
                        localStorage.setItem('autoRefreshDisabled', 'true');
                    }} catch (e) {{
                        // Local storage might not be available, proceed without it
                    }}
                }} else {{
                    // Start auto-refresh
                    refreshTimer = setTimeout(function() {{
                        window.location.reload();
                    }}, {auto_refresh_interval * 1000}); // {auto_refresh_interval} seconds
                    autoRefreshEnabled = true;
                    
                    // Update the UI
                    document.getElementById('refresh-status').innerText = 'Page refreshes every {auto_refresh_interval} seconds';
                    document.getElementById('stop-refresh-btn').innerText = 'Stop Auto-Refresh';
                    document.getElementById('stop-refresh-btn').style.backgroundColor = '#ff6b6b'; // Red for Stop
                    
                    // Save the preference in localStorage
                    try {{
                        localStorage.setItem('autoRefreshDisabled', 'false');
                    }} catch (e) {{
                        // Local storage might not be available, proceed without it
                    }}
                }}
                
                return false; // Prevent form submission
            }}
            
            // Check for previously saved preference
            document.addEventListener('DOMContentLoaded', function() {{
                try {{
                    if (localStorage.getItem('autoRefreshDisabled') === 'true') {{
                        // Auto-refresh was previously disabled
                        clearTimeout(refreshTimer);
                        autoRefreshEnabled = false;
                        document.getElementById('refresh-status').innerText = 'Auto-refresh disabled';
                        document.getElementById('stop-refresh-btn').innerText = 'Start Auto-Refresh';
                        document.getElementById('stop-refresh-btn').style.backgroundColor = '#4caf50'; // Green for Start
                    }}
                }} catch (e) {{
                    // Local storage might not be available, proceed without it
                }}
                
                // Simple JavaScript for filtering
                const filterButtons = document.querySelectorAll('.filter-btn');
                
                filterButtons.forEach(btn => {{
                    btn.addEventListener('click', function() {{
                        const filter = this.getAttribute('data-filter');
                        
                        // Toggle active class
                        filterButtons.forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                        
                        // Hide/show sections based on filter
                        if (filter === 'all') {{
                            document.querySelectorAll('.folder-section, .camera-section').forEach(
                                section => section.style.display = 'block'
                            );
                        }} else if (filter === 'folders') {{
                            document.querySelectorAll('.folder-section').forEach(
                                section => section.style.display = 'block'
                            );
                            document.querySelectorAll('.camera-section').forEach(
                                section => section.style.display = 'none'
                            );
                        }} else if (filter === 'cameras') {{
                            document.querySelectorAll('.folder-section').forEach(
                                section => section.style.display = 'none'
                            );
                            document.querySelectorAll('.camera-section').forEach(
                                section => section.style.display = 'block'
                            );
                        }}
                    }});
                }});
            }});
        </script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
            h3 {{
                color: #777;
                margin-top: 20px;
            }}
            .video-container {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 15px;
            }}
            .thumbnails {{
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
                flex-wrap: wrap;
            }}
            .thumbnail {{
                margin: 5px;
                border: 1px solid #ddd;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                color: #888;
                font-size: 0.9em;
            }}
            .folder-path {{
                font-family: monospace;
                background-color: #eee;
                padding: 5px;
                border-radius: 3px;
                font-size: 0.9em;
                overflow-wrap: break-word;
            }}
            .video-link {{
                display: block;
                margin-top: 5px;
                font-family: monospace;
                color: #0066cc;
                text-decoration: none;
            }}
            .video-link:hover {{
                text-decoration: underline;
            }}
            .summary {{
                background-color: #e9f7fe;
                border-left: 4px solid #1e88e5;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 3px;
            }}
            .filters {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
            }}
            .filter-btn {{
                padding: 8px 16px;
                background-color: #f1f1f1;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            .filter-btn:hover {{
                background-color: #e1e1e1;
            }}
            .filter-btn.active {{
                background-color: #1e88e5;
                color: white;
            }}
            #toc {{
                background-color: #fff;
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .toc-title {{
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .toc-list {{
                list-style-type: none;
                padding-left: 10px;
            }}
            .toc-list li {{
                margin-bottom: 5px;
            }}
            .toc-list a {{
                text-decoration: none;
                color: #0066cc;
            }}
            .toc-list a:hover {{
                text-decoration: underline;
            }}
            .progress-info {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 3px;
                font-style: italic;
            }}
            .control-button {{
                padding: 10px 15px;
                background-color: #ff6b6b;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 10px;
                font-weight: bold;
            }}
            .control-button:hover {{
                background-color: #ff5252;
            }}
            #controls {{
                margin-bottom: 20px;
            }}
            .note {{
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 3px;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <h1>Tesla Sentry Mode Motion Detection Results</h1>
        
        <div id="controls">
            <button id="stop-refresh-btn" class="control-button" onclick="stopAutoRefresh()">Stop Auto-Refresh</button>
            <span id="refresh-status">Page refreshes every {auto_refresh_interval} seconds</span>
        </div>
        
        <div class="note">
            This report uses relative paths and should be kept in this folder to maintain working video links.
        </div>
    """
    
    # Get progress tracking variables from global state if available
    try:
        from __main__ import progress
        total_processed = progress.completed if hasattr(progress, 'completed') else 0
        total_videos = progress.total_videos if hasattr(progress, 'total_videos') else 0
        status_text = f"{status} ({total_processed}/{total_videos} videos)" if status == "In Progress" and total_videos > 0 else status
    except:
        # If main progress tracker isn't available, just show status without counts
        status_text = status
    
    # Summary section
    components['summary'] = f"""
    <div class="summary">
        <p>Status: <b>{status_text}</b></p>
        <p>Last updated: <b>{last_updated}</b></p>
        <p>Total folders: <b>{len(sorted_folders)}</b></p>
        <p>Total cameras: <b>{len(sorted_cameras)}</b></p>
        <p>Total videos with motion: <b>{len(motion_videos)}</b></p>
        <p>Report generated: <b>{generation_time}</b></p>
    </div>
    """
    
    # Filters section (if needed)
    components['filters'] = ""
    if group_by == 'both':
        components['filters'] = """
        <div class="filters">
            <button class="filter-btn active" data-filter="all">Show All</button>
            <button class="filter-btn" data-filter="folders">Group by Folder</button>
            <button class="filter-btn" data-filter="cameras">Group by Camera</button>
        </div>
        """
    
    # Table of contents
    components['toc'] = '<div id="toc">\n<div class="toc-title">Table of Contents</div>\n<ul class="toc-list">\n'
    
    # Add folder links to TOC if grouping by folder
    if group_by in ['both', 'folder'] and sorted_folders:
        components['toc'] += '<li><strong>Folders</strong></li>\n'
        for folder in sorted_folders:
            folder_id = 'folder-' + str(hash(folder) % 10000000)
            components['toc'] += f'<li><a href="#{folder_id}">{os.path.basename(folder) or folder}</a></li>\n'
    
    # Add camera links to TOC if grouping by camera
    if group_by in ['both', 'camera'] and sorted_cameras:
        components['toc'] += '<li><strong>Cameras</strong></li>\n'
        for camera in sorted_cameras:
            camera_id = 'camera-' + camera.replace('_', '-')
            components['toc'] += f'<li><a href="#{camera_id}">{camera}</a></li>\n'
    
    components['toc'] += '</ul>\n</div>\n'
    
    # Close HTML body
    components['close'] = """
    </body>
    </html>
    """
    
    return components

def main():
    global abort_requested
    
    try:
        # Register the signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        overall_start_time = time.time()
        
        # Setup command-line argument parsing
        parser = argparse.ArgumentParser(
            description='Find videos with substantial motion in a directory and its subdirectories'
        )
        parser.add_argument('root_path', help='Root directory to search for videos')
        parser.add_argument('--extension', default='.mp4', help='Video file extension to search for (default: .mp4)')
        parser.add_argument('--threshold', type=int, default=45, help='Threshold for motion detection (0-255, default: 45)')
        parser.add_argument('--min-area', type=float, default=2.5, 
                            help='Minimum percentage of frame that needs to show motion (default: 2.5)')
        parser.add_argument('--workers', type=int, default=8, help='Maximum number of parallel workers (default: 8)')
        parser.add_argument('--output', default='motion_results.html', help='Output file name for the HTML report (default: motion_results.html in root_path)')
        parser.add_argument('--sample-frames', action='store_true', default=True, help='Sample frames instead of processing all frames (default: True)')
        parser.add_argument('--no-sample-frames', action='store_false', dest='sample_frames', help='Process all frames instead of sampling')
        parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU acceleration if available (default: True)')
        parser.add_argument('--no-gpu', action='store_false', dest='use_gpu', help='Disable GPU acceleration')
        parser.add_argument('--no-thumbnails', action='store_false', dest='generate_thumbnails', default=True, help='Disable thumbnail generation')
        parser.add_argument('--filter-camera', nargs='+', choices=['front', 'back', 'left_repeater', 'right_repeater'], 
                           help='Filter by camera type(s). Can specify multiple types separated by spaces.')
        parser.add_argument('--group-by', choices=['folder', 'camera', 'both'], default='both', 
                            help='How to group results: by folder, by camera type, or both (default: both)')
        parser.add_argument('--no-realtime', action='store_false', dest='realtime_updates', default=True, 
                            help='Disable real-time updates to the HTML report')
        parser.add_argument('--verbose-logging', action='store_true', default=False, 
                            help='Enable detailed debugging information in log file')
        parser.add_argument('--resume', action='store_true', default=False, 
                            help='Resume processing by skipping videos already in the HTML report')
        parser.add_argument('--limit', type=int, default=0, 
                            help='Limit the number of videos to process (0 means no limit)')
        
        args = parser.parse_args()
        
        # Set up logging based on verbosity
        if args.verbose_logging:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled")
        
        # Log the command line arguments
        logger.info(f"Starting motion detection with args: {vars(args)}")
        
        # Check if the directory exists
        if not os.path.isdir(args.root_path):
            logger.error(f"Directory '{args.root_path}' does not exist.")
            if USE_RICH:
                Console().print(f"[{get_timestamp()}] Error: The directory '{args.root_path}' does not exist.", style="red")
            else:
                print(f"[{get_timestamp()}] Error: The directory '{args.root_path}' does not exist.")
            return
        
        # Print the configuration
        print_configuration(args)
        
        if USE_RICH:
            Console().print(f"[{get_timestamp()}] Searching for videos with motion in {args.root_path}...")
        else:
            print(f"[{get_timestamp()}] Searching for videos with motion in {args.root_path}...")
        
        # Create the full path for the HTML output in the root directory
        root_path = os.path.abspath(args.root_path)
        html_filename = os.path.basename(args.output)
        html_filepath = os.path.join(root_path, html_filename)
        
        if USE_RICH:
            Console().print(f"[{get_timestamp()}] HTML report will be saved to: {html_filepath}", style="green")
        else:
            print(f"[{get_timestamp()}] HTML report will be saved to: {html_filepath}")
        
        # Get list of already processed videos if resuming
        already_processed = set()
        if args.resume and os.path.exists(html_filepath):
            try:
                logger.info(f"Resuming processing - reading existing report: {html_filepath}")
                with open(html_filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Extract all video paths from the HTML
                # Now we need to handle both absolute and relative paths
                abs_video_paths = re.findall(r'<a href="file://([^"]+)"', html_content)
                rel_video_paths = re.findall(r'<a href="([^"]+)" class="video-link">', html_content)
                
                # Convert relative paths to absolute
                for rel_path in rel_video_paths:
                    if not rel_path.startswith('file://'):
                        abs_path = os.path.abspath(os.path.join(root_path, rel_path))
                        already_processed.add(abs_path)
                
                # Add absolute paths directly
                already_processed.update(abs_video_paths)
                
                logger.info(f"Found {len(already_processed)} already processed videos in existing report")
                if USE_RICH:
                    Console().print(f"[{get_timestamp()}] Resuming from existing report - {len(already_processed)} videos already processed", style="green")
                else:
                    print(f"[{get_timestamp()}] Resuming from existing report - {len(already_processed)} videos already processed")
            except Exception as e:
                logger.error(f"Error parsing existing report for resume: {e}")
                already_processed = set()
        
        # Initialize the HTML progress writer if real-time updates are enabled
        html_writer = None
        if args.realtime_updates:
            html_writer = HTMLProgressWriter(html_filepath, args.group_by, root_path)
            if USE_RICH:
                Console().print(f"[{get_timestamp()}] Real-time HTML updates enabled", style="green")
            else:
                print(f"[{get_timestamp()}] Real-time HTML updates enabled")
        
        # Find videos with motion
        motion_videos = find_motion_videos(
            args.root_path, 
            args.extension, 
            args.threshold, 
            args.min_area,
            args.workers,
            args.sample_frames,
            args.use_gpu,
            html_writer,
            already_processed,
            args.limit,
            args.filter_camera
        )
        
        # If aborted, exit here
        if abort_requested:
            if html_writer:
                html_writer.update_html()  # Final update before exiting
                
            logger.info("Processing aborted, exiting without final report.")
            total_time = time.time() - overall_start_time
            
            if USE_RICH:
                Console().print(f"[{get_timestamp()}] Abort completed. Found {len(motion_videos)} videos with motion in {format_time(total_time)}.", style="yellow")
            else:
                print(f"[{get_timestamp()}] Abort completed. Found {len(motion_videos)} videos with motion in {format_time(total_time)}.")
            return
        
        # Filter by camera type if specified
        if args.filter_camera and motion_videos:
            filtered_videos = []
            for video in motion_videos:
                camera_type = extract_camera_type(video)
                if camera_type == args.filter_camera:
                    filtered_videos.append(video)
            
            logger.info(f"Filtered to {len(filtered_videos)}/{len(motion_videos)} videos with camera type '{args.filter_camera}'")
            if USE_RICH:
                Console().print(f"[{get_timestamp()}] Filtered to {len(filtered_videos)} videos with camera type '{args.filter_camera}'")
            else:
                print(f"[{get_timestamp()}] Filtered to {len(filtered_videos)} videos with camera type '{args.filter_camera}'")
            
            motion_videos = filtered_videos
        
        # Save results to HTML file
        if motion_videos:
            # Generate HTML report
            if USE_RICH:
                Console().print(f"[{get_timestamp()}] Generating final HTML report with {len(motion_videos)} videos...")
            else:
                print(f"[{get_timestamp()}] Generating final HTML report with {len(motion_videos)} videos...")
            
            # If using real-time updates, just finalize the existing report
            if html_writer:
                if html_writer.finalize():
                    if USE_RICH:
                        Console().print(f"[{get_timestamp()}] Final HTML report saved to: {html_filepath}", style="green")
                    else:
                        print(f"[{get_timestamp()}] Final HTML report saved to: {html_filepath}")
                else:
                    if USE_RICH:
                        Console().print(f"[{get_timestamp()}] Error finalizing HTML report", style="red")
                    else:
                        print(f"[{get_timestamp()}] Error finalizing HTML report")
            else:
                # If not using real-time updates, generate the entire report now
                if generate_html_report(motion_videos, html_filepath, args.group_by, root_path):
                    if USE_RICH:
                        Console().print(f"[{get_timestamp()}] HTML report saved to: {html_filepath}", style="green")
                    else:
                        print(f"[{get_timestamp()}] HTML report saved to: {html_filepath}")
                else:
                    # Try desktop as fallback 
                    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
                    desktop_file = os.path.join(desktop_path, html_filename)
                    if generate_html_report(motion_videos, desktop_file, args.group_by, root_path):
                        if USE_RICH:
                            Console().print(f"[{get_timestamp()}] HTML report saved to Desktop: {desktop_file}", style="green")
                        else:
                            print(f"[{get_timestamp()}] HTML report saved to Desktop: {desktop_file}")
        
        # Log completion
        total_time = time.time() - overall_start_time
        logger.info(f"Total execution time: {format_time(total_time)} - Found {len(motion_videos)} videos with motion")
        
        # Print total execution time
        if USE_RICH:
            Console().print(f"[{get_timestamp()}] Total execution time: {format_time(total_time)}")
        else:
            print(f"[{get_timestamp()}] Total execution time: {format_time(total_time)}")
    
    except KeyboardInterrupt:
        # Already handled by signal handler
        pass
    
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        if USE_RICH:
            Console().print(f"[{get_timestamp()}] Error: {e}", style="red")
        else:
            print(f"[{get_timestamp()}] Error: {e}")

if __name__ == "__main__":
    main() 