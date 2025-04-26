import gradio as gr
import time
import datetime
import random
import os
from typing import List, Dict, Any, Optional

from modules.video_queue import JobStatus
from modules.prompt_handler import get_section_boundaries, get_quick_prompts
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html


def create_interface(
    process_fn, 
    monitor_fn, 
    end_process_fn, 
    update_queue_status_fn,
    load_lora_file_fn, 
    default_prompt: str = 'The girl dances gracefully, with clear movements, full of charm.',
    lora_names: list = [],
    lora_values: list = []
):
    """
    Create the Gradio interface for the video generation application
    
    Args:
        process_fn: Function to process a new job
        monitor_fn: Function to monitor an existing job
        end_process_fn: Function to cancel the current job
        update_queue_status_fn: Function to update the queue status display
        default_prompt: Default prompt text
        lora_names: List of loaded LoRA names
        
    Returns:
        Gradio Blocks interface
    """
    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()
    
    # Create the interface
    css = make_progress_bar_css()
    css += """
    .contain-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        background: #222;
    }
    """

    block = gr.Blocks(css=css, theme="soft").queue()
    
    with block:
        gr.Markdown('# FramePack Studio')
        
        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            sources='upload',
                            type="numpy",
                            label="Image (optional)",
                            height=420,
                            elem_classes="contain-image"
                        )
                        
                        prompt = gr.Textbox(label="Prompt", value=default_prompt)

                        #example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                        #example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)
                        
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                json_upload = gr.File(
                                    label="Upload Metadata JSON (optional)",
                                    file_types=[".json"],
                                    type="filepath",
                                    height=150,
                                )
                                save_metadata = gr.Checkbox(label="Save Metadata", value=True, info="Store prompt/seed in output image metadata. Saves all generation params in a JSON file with the same name")   
                            with gr.Row():
                                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                            
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=31337, precision=0)
                                randomize_seed = gr.Checkbox(label="Randomize", value=False, info="Generate a new random seed for each job")
                            with gr.Row("LoRA"):
                                for lora in lora_names:
                                    lora_values.append(gr.Slider(label=lora, minimum=0.0, maximum=2.0, value=1.0, step=0.01,))    
                            total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                            steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')


                        with gr.Accordion("Advanced Parameters", open=False):    
                            latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                            gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change
                            mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=0, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                            gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                           





                                

                        with gr.Row():
                            start_button = gr.Button(value="Add to Queue")
                            # Removed the monitor button since we'll auto-monitor
                            
                    with gr.Column():
                        preview_image = gr.Image(label="Next Latents", height=150, visible=True, type="numpy")
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                        #gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')

                        with gr.Row():  
                            current_job_id = gr.Textbox(label="Current Job ID", visible=True, interactive=True) 
                            end_button = gr.Button(value="Cancel Current Job", interactive=True) 
                        with gr.Row():     
                            queue_status = gr.DataFrame(
                                headers=["Job ID", "Status", "Created", "Started", "Completed", "Elapsed"],
                                datatype=["str", "str", "str", "str", "str", "str"],
                                label="Job Queue"
                            )

            with gr.TabItem("LoRA Management"):
                gr.Markdown("## Manage LoRA Models")
                
                with gr.Row():
                    lora_upload = gr.File(
                        label="Upload LoRA File (.safetensors)",
                        file_types=[".safetensors"],
                        type="filepath"
                    )
                    
                    lora_upload_status = gr.Markdown("")
                
                with gr.Row():
                    lora_list = gr.Dataframe(
                        headers=["Name", "Path", "Status"],
                        datatype=["str", "str", "str"],
                        label="Installed LoRAs"
                    )
                    
                    refresh_lora_btn = gr.Button(value="Refresh List")
                
                # Function to list all LoRAs in the directory
                def list_loras():
                    lora_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'loras')
                    if not os.path.exists(lora_dir):
                        os.makedirs(lora_dir, exist_ok=True)
                        
                    loras = []
                    for file in os.listdir(lora_dir):
                        if file.endswith('.safetensors'):
                            loras.append([
                                file.split('.')[0],  # Name without extension
                                os.path.join(lora_dir, file),  # Full path
                                "Loaded" if file.split('.')[0] in lora_names else "Not Loaded"  # Status
                            ])
                    return loras
                
                # Connect the refresh button
                refresh_lora_btn.click(
                    fn=list_loras,
                    outputs=[lora_list]
                )
                
                # Initial load of LoRA list
                lora_list.value = list_loras()
                        
        # Add a refresh timer that updates the queue status every 2 seconds
        refresh_timer = gr.Number(value=0, visible=False)
        
        def refresh_timer_fn():
            """Updates the timer value periodically to trigger queue refresh"""
            return int(time.time())
        
        # Function to handle randomizing the seed if checkbox is checked
        def process_with_random_seed(*args):
            # Extract all arguments
            input_image, prompt_text, n_prompt, seed_value, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, randomize_seed_checked, save_metadata_checked, *lora_args = args
            
            # If randomize seed is checked, generate a new random seed
            if randomize_seed_checked:
                seed_value = random.randint(0, 2147483647)  # Max 32-bit integer
                print(f"Randomized seed: {seed_value}")
            
            # Call the original process function with the potentially updated seed
            return process_fn(input_image, prompt_text, n_prompt, seed_value, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, save_metadata_checked, *lora_args)
            
        # Connect the main process function
        ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, randomize_seed, save_metadata]
        
        # Add LoRA values to inputs if any exist
        if lora_values:
            ips.extend(lora_values)
        
        # Modified process function that updates the queue status after adding a job
        def process_with_queue_update(*args):
            # Call the process function with random seed handling
            result = process_with_random_seed(*args)
            
            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                queue_status_data = update_queue_status_fn()
                return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data]
            return result + [update_queue_status_fn()]
            
        # Custom end process function that ensures the queue is updated
        def end_process_with_update():
            queue_status_data = end_process_fn()
            # Make sure to return the queue status data
            return queue_status_data

        # Load that metadata!
        def load_metadata_from_json(json_path):
            if not json_path:
                return [gr.update(), gr.update()]
            
            try:
                import json
                
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                prompt = metadata.get('prompt')
                seed = metadata.get('seed')
                
                # Check for LoRA values in metadata
                lora_values = metadata.get('lora_values', {})
                
                print(f"Loaded metadata from JSON: {json_path}")
                print(f"Prompt: {prompt}, Seed: {seed}")
                
                # Update the UI components
                updates = [
                    gr.update(value=prompt) if prompt else gr.update(),
                    gr.update(value=seed) if seed is not None else gr.update()
                ]
                
                # Update LoRA sliders if they exist in metadata
                if lora_values:
                    for lora_name, lora_value in lora_values.items():
                        # Find the corresponding slider
                        for i, lora in enumerate(lora_names):
                            if lora == lora_name:
                                updates.append(gr.update(value=lora_value))
                
                return updates
                
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return [gr.update(), gr.update()]
        
        # Function to handle LoRA file upload
        def handle_lora_upload(file):
            if not file:
                return gr.update(), "No file uploaded"
            
            try:
                # Get the filename from the path
                _, lora_name = os.path.split(file.name)
                
                # Copy the file to the lora directory
                lora_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'loras')
                os.makedirs(lora_dir, exist_ok=True)
                
                lora_dest = os.path.join(lora_dir, lora_name)
                import shutil
                shutil.copy(file.name, lora_dest)
                
                return gr.update(), f"Successfully uploaded {lora_name}"
            except Exception as e:
                return gr.update(), f"Error uploading LoRA: {e}"
            
        # Connect the buttons to their respective functions
        start_button.click(
            fn=process_with_queue_update, 
            inputs=ips, 
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status]
        )
        
        # Connect the end button to cancel the current job and update the queue
        end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status]
        )
        
        # Auto-monitor the current job when job_id changes
        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button]
        )
        
        # Set up auto-refresh for queue status
        refresh_timer.change(
            fn=update_queue_status_fn,
            outputs=[queue_status]
        )
        
        # Connect LoRA file uploader
        # lora_file.change(
        #     fn=load_lora_file_fn, 
        #     inputs=[lora_file],
        #     outputs=[available_loras, lora_error]
        # )
        
        # Connect LoRA upload in management tab
        lora_upload.change(
            fn=handle_lora_upload,
            inputs=[lora_upload],
            outputs=[lora_upload, lora_upload_status]
        )
        
        # Connect JSON metadata loader
        json_upload.change(
            fn=load_metadata_from_json,
            inputs=[json_upload],
            outputs=[prompt, seed] + lora_values
        )

        # Create a timer event every 2 seconds
        def start_refresh_timer():
            """Function to start a thread that updates the queue status periodically"""
            import threading
            
            def refresh_loop():
                while True:
                    # Sleep for 2 seconds
                    time.sleep(2)
                    # Update the timer value to trigger the queue refresh
                    # We need to use .update() instead of direct assignment to trigger Gradio's update
                    try:
                        refresh_timer.update(int(time.time()))
                    except:
                        # If the interface is closed, this will throw an exception
                        break
            
            # Start the refresh thread
            thread = threading.Thread(target=refresh_loop, daemon=True)
            thread.start()

        # Then after defining your interface but before returning it:
        start_refresh_timer()
            
    return block


def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

        # Calculate elapsed time
        elapsed_time = ""
        if job.started_at:
            if job.completed_at:
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
                elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s"
            else:
                # For running jobs, calculate elapsed time from now
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                current_datetime = datetime.datetime.now()
                elapsed_seconds = (current_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s (running)"

        position = job.queue_position if hasattr(job, 'queue_position') else ""

        rows.append([
            job.id[:6] + '...',
            job.status.value,
            created,
            started,
            completed,
            elapsed_time
        ])
    return rows
