import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import queue as queue_module  # Renamed to avoid conflicts

from diffusers_helper.thread_utils import AsyncStream


# Simple LIFO queue implementation to avoid dependency on queue.LifoQueue
class SimpleLifoQueue:
    def __init__(self):
        self._queue = []
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
    
    def put(self, item):
        with self._mutex:
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self):
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            return self._queue.pop()
    
    def task_done(self):
        pass  # For compatibility with queue.Queue


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None  # Path to output video
    error: Optional[str] = None
    progress_data: Dict[str, Any] = field(default_factory=dict)  # Store latest progress data
    stream: Optional[AsyncStream] = None  # Stream for this job
    queue_position: Optional[int] = None  # Position in queue for display


class VideoJobQueue:
    def __init__(self):
        self.queue = queue_module.Queue()  # Using standard Queue instead of LifoQueue
        self.jobs = {}
        self.current_job = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.worker_function = None  # Will be set from outside
        self.is_processing = False  # Flag to track if we're currently processing a job
    
    def set_worker_function(self, worker_function):
        """Set the worker function to use for processing jobs"""
        self.worker_function = worker_function
    
    def add_job(self, params):
        """Add a job to the queue and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            params=params,
            status=JobStatus.PENDING,
            created_at=time.time(),
            progress_data={},
            stream=AsyncStream()
        )
        
        with self.lock:
            print(f"Adding job {job_id} to queue, current job is {self.current_job.id if self.current_job else 'None'}")
            self.jobs[job_id] = job
            self.queue.put(job_id)
        
        return job_id
    
    def get_job(self, job_id):
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self):
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def cancel_job(self, job_id):
        """Cancel a pending job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                return True
            elif job and job.status == JobStatus.RUNNING:
                # Send cancel signal to the job's stream
                job.stream.input_queue.push('end')
                # Mark job as cancelled (this will be confirmed when the worker processes the end signal)
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                return True
            return False
    
    def get_queue_position(self, job_id):
        """Get position in queue (0 = currently running)"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
                
            if job.status == JobStatus.RUNNING:
                return 0
                
            if job.status != JobStatus.PENDING:
                return None
                
            # Count pending jobs ahead in queue
            position = 1  # Start at 1 because 0 means running
            for j in self.jobs.values():
                if (j.status == JobStatus.PENDING and 
                    j.created_at < job.created_at):
                    position += 1
            return position
    
    def update_job_progress(self, job_id, progress_data):
        """Update job progress data"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress_data = progress_data
    
    def _worker_loop(self):
        """Worker thread that processes jobs from the queue"""
        while True:
            try:
                # Get the next job ID from the queue
                try:
                    job_id = self.queue.get(block=True, timeout=1.0)  # Use timeout to allow periodic checks
                except queue_module.Empty:
                    # No jobs in queue, just continue the loop
                    continue
                
                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        self.queue.task_done()
                        continue
                    
                    # Skip cancelled jobs
                    if job.status == JobStatus.CANCELLED:
                        self.queue.task_done()
                        continue
                    
                    # If we're already processing a job, wait for it to complete
                    if self.is_processing:
                        # Put the job back in the queue
                        self.queue.put(job_id)
                        self.queue.task_done()
                        time.sleep(0.1)  # Small delay to prevent busy waiting
                        continue
                    
                    print(f"Starting job {job_id}, current job was {self.current_job.id if self.current_job else 'None'}")
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    self.current_job = job
                    self.is_processing = True
                
                job_completed = False
                
                try:
                    if self.worker_function is None:
                        raise ValueError("Worker function not set. Call set_worker_function() first.")
                    
                    # Start the worker function with the job parameters
                    from diffusers_helper.thread_utils import async_run
                    async_run(
                        self.worker_function,
                        **job.params,
                        job_stream=job.stream
                    )
                    
                    # Process the results from the stream
                    output_filename = None
                    
                    # Set a maximum time to wait for the job to complete
                    max_wait_time = 3600  # 1 hour in seconds
                    start_time = time.time()
                    last_activity_time = time.time()
                    
                    while True:
                        # Check if job has been cancelled before processing next output
                        with self.lock:
                            if job.status == JobStatus.CANCELLED:
                                print(f"Job {job_id} was cancelled, breaking out of processing loop")
                                job_completed = True
                                break
                        
                        # Check if we've been waiting too long without any activity
                        current_time = time.time()
                        if current_time - start_time > max_wait_time:
                            print(f"Job {job_id} timed out after {max_wait_time} seconds")
                            with self.lock:
                                job.status = JobStatus.FAILED
                                job.error = "Job timed out"
                                job.completed_at = time.time()
                            job_completed = True
                            break
                        
                        # Check for inactivity (no output for a while)
                        if current_time - last_activity_time > 60:  # 1 minute of inactivity
                            print(f"Checking if job {job_id} is still active...")
                            # Just a periodic check, don't break yet
                        
                        try:
                            # Try to get data from the queue with a non-blocking approach
                            flag, data = job.stream.output_queue.next()
                            
                            # Update activity time since we got some data
                            last_activity_time = time.time()
                            
                            if flag == 'file':
                                output_filename = data
                                with self.lock:
                                    job.result = output_filename
                            
                            elif flag == 'progress':
                                preview, desc, html = data
                                with self.lock:
                                    job.progress_data = {
                                        'preview': preview,
                                        'desc': desc,
                                        'html': html
                                    }
                            
                            elif flag == 'end':
                                print(f"Received end signal for job {job_id}")
                                job_completed = True
                                break
                                
                        except IndexError:
                            # Queue is empty, wait a bit and try again
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            print(f"Error processing job output: {e}")
                            # Wait a bit before trying again
                            time.sleep(0.1)
                            continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing job {job_id}: {e}")
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.completed_at = time.time()
                    job_completed = True
                
                finally:
                    with self.lock:
                        # Make sure we properly clean up the job state
                        if job.status == JobStatus.RUNNING:
                            if job_completed:
                                job.status = JobStatus.COMPLETED
                            else:
                                # Something went wrong but we didn't mark it as completed
                                job.status = JobStatus.FAILED
                                job.error = "Job processing was interrupted"
                            
                            job.completed_at = time.time()
                        
                        print(f"Finishing job {job_id} with status {job.status}")
                        self.is_processing = False
                        self.current_job = None
                        self.queue.task_done()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in worker loop: {e}")
                
                # Make sure we reset processing state if there was an error
                with self.lock:
                    self.is_processing = False
                    if self.current_job:
                        self.current_job.status = JobStatus.FAILED
                        self.current_job.error = f"Worker loop error: {str(e)}"
                        self.current_job.completed_at = time.time()
                        self.current_job = None
                
                time.sleep(0.5)  # Prevent tight loop on error
