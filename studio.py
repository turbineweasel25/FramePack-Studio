from diffusers_helper.hf_login import login

import asyncio
import json
import os
import time
import argparse
import traceback
import einops
import numpy as np
import torch

#Hacky fix for socket timeout in Windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper import lora_utils

# Import from modules
from modules.video_queue import VideoJobQueue, JobStatus
from modules.prompt_handler import parse_timestamped_prompt
from modules.interface import create_interface, format_queue_status


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lora", type=str, default=None, help="Lora path (comma separated for multiple)")
args = parser.parse_args()

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load models
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

# Configure models
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# Create lora directory if it doesn't exist
lora_dir = os.path.join(os.path.dirname(__file__), 'loras')
os.makedirs(lora_dir, exist_ok=True)

# Initialize LoRA support
lora_names = []
lora_values = []

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define LoRA folder path relative to the script directory
lora_folder = os.path.join(script_dir, "loras")

if os.path.isdir(lora_folder):
    # Get all files with .safetensors or other LoRA extensions
    lora_files = [f for f in os.listdir(lora_folder) 
                 if f.endswith('.safetensors') or f.endswith('.pt')]
    
    for lora_file in lora_files:
        print(f"Loading lora {lora_file}")
        transformer = lora_utils.load_lora(transformer, lora_folder, lora_file)
        lora_names.append(lora_file.split('.')[0])
    
    if not lora_files:
        print(f"No LoRA files found in {lora_folder}")
else:
    print(f"LoRA folder {lora_folder} does not exist")

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Create job queue
job_queue = VideoJobQueue()


def move_lora_adapters_to_device(model, target_device):
    """
    Move all LoRA adapters in a model to the specified device.
    This handles the PEFT implementation of LoRA.
    """
    print(f"Moving all LoRA adapters to {target_device}")
    
    # First, find all modules with LoRA adapters
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'active_adapter') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules.append((name, module))
    
    # Now move all LoRA components to the target device
    for name, module in lora_modules:
        # Get the active adapter name
        active_adapter = module.active_adapter
        
        # Move the LoRA layers to the target device
        if active_adapter is not None:
            if isinstance(module.lora_A, torch.nn.ModuleDict):
                # Handle ModuleDict case (PEFT implementation)
                for adapter_name in list(module.lora_A.keys()):
                    # Move lora_A
                    if adapter_name in module.lora_A:
                        module.lora_A[adapter_name] = module.lora_A[adapter_name].to(target_device)
                    
                    # Move lora_B
                    if adapter_name in module.lora_B:
                        module.lora_B[adapter_name] = module.lora_B[adapter_name].to(target_device)
                    
                    # Move scaling
                    if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and adapter_name in module.scaling:
                        if isinstance(module.scaling[adapter_name], torch.Tensor):
                            module.scaling[adapter_name] = module.scaling[adapter_name].to(target_device)
            else:
                # Handle direct attribute case
                if hasattr(module, 'lora_A') and module.lora_A is not None:
                    module.lora_A = module.lora_A.to(target_device)
                if hasattr(module, 'lora_B') and module.lora_B is not None:
                    module.lora_B = module.lora_B.to(target_device)
                if hasattr(module, 'scaling') and module.scaling is not None:
                    if isinstance(module.scaling, torch.Tensor):
                        module.scaling = module.scaling.to(target_device)
    
    print(f"Moved all LoRA adapters to {target_device}")
    return model


# Function to load a LoRA file
def load_lora_file(lora_file):
    if not lora_file:
        return None, "No file selected"
    
    try:
        # Get the filename from the path
        _, lora_name = os.path.split(lora_file)
        
        # Copy the file to the lora directory
        lora_dest = os.path.join(lora_dir, lora_name)
        import shutil
        shutil.copy(lora_file, lora_dest)
        
        # Load the LoRA
        global transformer, lora_names
        transformer = lora_utils.load_lora(transformer, lora_dir, lora_name)
        
        # Add to lora_names if not already there
        lora_base_name = lora_name.split('.')[0]
        if lora_base_name not in lora_names:
            lora_names.append(lora_base_name)
        
        # Get the current device of the transformer
        device = next(transformer.parameters()).device
        
        # Move all LoRA adapters to the same device as the base model
        move_lora_adapters_to_device(transformer, device)
        
        print(f"Loaded LoRA: {lora_name}")
        return gr.update(choices=lora_names), f"Successfully loaded LoRA: {lora_name}"
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return None, f"Error loading LoRA: {e}"


@torch.no_grad()
def worker(input_image, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, save_metadata, lora_values=None, job_stream=None):
    # Use the provided job_stream or the global stream
    stream_to_use = job_stream if job_stream is not None else stream

    print(f"Worker received lora_values: {lora_values}, type: {type(lora_values)}")
    if lora_values and isinstance(lora_values, tuple) and len(lora_values) > 0:
        print(f"First lora value: {lora_values[0]}, type: {type(lora_values[0])}")
    
    # Apply LoRA weights if any are provided
    if lora_names and lora_values:
        print("setting loras", lora_names, lora_values)
        
        # Set adapters
        lora_utils.set_adapters(transformer, lora_names, lora_values)
    
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Parse the timestamped prompt with boundary snapping and reversing
    prompt_sections = parse_timestamped_prompt(prompt_text, total_second_length, latent_window_size)
    
    job_id = generate_timestamp()

    stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Pre-encode all prompts
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding all prompts...'))))
        
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        
        # Create a dictionary to store encoded prompts
        encoded_prompts = {}
        for section in prompt_sections:
            if section.prompt not in encoded_prompts:
                llama_vec, clip_l_pooler = encode_prompt_conds(
                    section.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                )
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                encoded_prompts[section.prompt] = (llama_vec, llama_attention_mask, clip_l_pooler)
        
        # Encode negative prompt
        if cfg == 1:
            llama_vec_n, llama_attention_mask_n, clip_l_pooler_n = (
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][0]),
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][1]),
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][2])
            )
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        if save_metadata:
            metadata = PngInfo()
            metadata.add_text("prompt", prompt_text)
            metadata.add_text("seed", str(seed))
            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'), pnginfo=metadata)
        
            #save in JSON too since gradio can't extract metadata from a PNG upload. Let's save some more stuff too.
            metadata_dict = {
                "prompt": prompt_text,
                "seed": seed,
                "total_second_length": total_second_length,
                "steps": steps,
                "cfg": cfg,
                "gs": gs,
                "rs": rs,
                "latent_window_size" : latent_window_size,
                "mp4_crf" : mp4_crf,
                "timestamp": time.time()
            }
            
            # Add LoRA values to metadata if any
            if lora_names and lora_values:
                lora_data = dict(zip(lora_names, lora_values))
                metadata_dict["lora_values"] = lora_data
                
            with open(os.path.join(outputs_folder, f'{job_id}.json'), 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        else:
            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            llama_vec = llama_vec.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            encoded_prompts[prompt_key] = (llama_vec, llama_attention_mask, clip_l_pooler)
            
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream_to_use.input_queue.top() == 'end':
                stream_to_use.output_queue.push(('end', None))
                return

           # Calculate current time position to determine which prompt to use
            current_time_position = (total_generated_latent_frames * 4 - 3) / 30  # in seconds
            if current_time_position < 0:
                current_time_position = 0.01

            
            # Find the appropriate prompt for this section
            current_prompt = prompt_sections[0].prompt  # Default to first prompt
            for section in prompt_sections:
                if section.start_time <= current_time_position and (section.end_time is None or current_time_position < section.end_time):
                    current_prompt = section.prompt
                    break
            
            # Get the encoded prompt for this section
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[current_prompt]

            # Calculate the original (non-reversed) time position for display
            original_time_position = total_second_length - current_time_position
            if original_time_position < 0:
                original_time_position = 0
                
            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, ' 
                  f'time position: {current_time_position:.2f}s (original: {original_time_position:.2f}s), '
                  f'using prompt: {current_prompt[:30]}...')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                
                # Move all LoRA adapters to the GPU
                if lora_names:
                    move_lora_adapters_to_device(transformer, gpu)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            # Before sampling, ensure all LoRA adapters are on the correct device
            if lora_names:
                device = next(transformer.parameters()).device
                print(f"Ensuring all LoRA adapters are on device {device}")
                move_lora_adapters_to_device(transformer, device)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream_to_use.input_queue.top() == 'end':
                    stream_to_use.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                
                # Calculate current time position and original (non-reversed) position
                current_pos = (total_generated_latent_frames * 4 - 3) / 30
                original_pos = total_second_length - current_pos
                if current_pos < 0: current_pos = 0
                if original_pos < 0: original_pos = 0
                
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, ' \
                       f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). ' \
                       f'Current position: {current_pos:.2f}s (original: {original_pos:.2f}s). ' \
                       f'Using prompt: "{current_prompt[:50]}..."'
                
                # Store progress data in a format that includes the preview image
                progress_data = {
                    'preview': preview,
                    'desc': desc,
                    'html': make_progress_bar_html(percentage, hint)
                }
                
                # Update the job's progress data in the queue
                if job_stream is not None:
                    job = job_queue.get_job(job_id)
                    if job:
                        job.progress_data = progress_data
                
                stream_to_use.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                # Before offloading, move LoRA adapters to CPU if they exist
                if lora_names:
                    move_lora_adapters_to_device(transformer, cpu)
                
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream_to_use.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream_to_use.output_queue.push(('end', None))
    return


# Set the worker function for the job queue
job_queue.set_worker_function(worker)


def process(input_image, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, save_metadata, *lora_values):
    
    # Create a blank black image if no input image is provided
    if input_image is None:
        # Create a default black image (640x640)
        default_height, default_width = 640, 640
        input_image = np.zeros((default_height, default_width, 3), dtype=np.uint8)
        print("No input image provided. Using a blank black image.")
    
    # Create job parameters
    job_params = {
        'input_image': input_image.copy(),  # Make a copy to avoid reference issues
        'prompt_text': prompt_text,
        'n_prompt': n_prompt,
        'seed': seed,
        'total_second_length': total_second_length,
        'latent_window_size': latent_window_size,
        'steps': steps,
        'cfg': cfg,
        'gs': gs,
        'rs': rs,
        'gpu_memory_preservation': gpu_memory_preservation,
        'use_teacache': use_teacache,
        'mp4_crf': mp4_crf,
        'save_metadata': save_metadata
    }
    
    # Add LoRA values if provided - extract them from the tuple
    if lora_values:
        # Convert tuple to list
        lora_values_list = list(lora_values)
        job_params['lora_values'] = lora_values_list
    
    # Add job to queue
    job_id = job_queue.add_job(job_params)
    print(f"Added job {job_id} to queue")
    
    queue_status = update_queue_status()
    # Return immediately after adding to queue
    return None, job_id, None, '', f'Job added to queue. Job ID: {job_id}', gr.update(interactive=True), gr.update(interactive=True)



def end_process():
    """Cancel the current running job and update the queue status"""
    print("Cancelling current job")
    with job_queue.lock:
        if job_queue.current_job:
            job_id = job_queue.current_job.id
            print(f"Cancelling job {job_id}")

            # Send the end signal to the job's stream
            if job_queue.current_job.stream:
                job_queue.current_job.stream.input_queue.push('end')
                
            # Mark the job as cancelled
            job_queue.current_job.status = JobStatus.CANCELLED
            job_queue.current_job.completed_at = time.time()  # Set completion time
    
    # Force an update to the queue status
    return update_queue_status()


def update_queue_status():
    """Update queue status and refresh job positions"""
    jobs = job_queue.get_all_jobs()
    for job in jobs:
        if job.status == JobStatus.PENDING:
            job.queue_position = job_queue.get_queue_position(job.id)
    
    # Make sure to update current running job info
    if job_queue.current_job:
        # Make sure the running job is showing status = RUNNING
        job_queue.current_job.status = JobStatus.RUNNING
    
    return format_queue_status(jobs)


def monitor_job(job_id):
    """Monitor a specific job with improved error handling"""
    if not job_id:
        return None, None, None, '', 'No job ID provided', gr.update(interactive=True), gr.update(interactive=True)
    
    job = job_queue.get_job(job_id)
    
    if not job:
        return None, None, None, '', 'Job not found', gr.update(interactive=True), gr.update(interactive=True)
 
    # Make sure preview is visible from the start
    yield None, job_id, gr.update(visible=True), '', 'Initializing job...', gr.update(interactive=True), gr.update(interactive=True)
    
    while True:
        job = job_queue.get_job(job_id)
        
        if not job:
            return None, None, None, '', 'Job not found', gr.update(interactive=True), gr.update(interactive=True)
        
        if job.status == JobStatus.PENDING:
            position = job_queue.get_queue_position(job_id)
            yield None, job_id, gr.update(visible=True), '', f'Waiting in queue. Position: {position}', gr.update(interactive=True), gr.update(interactive=True)
        
        elif job.status == JobStatus.RUNNING:
            if job.progress_data and 'preview' in job.progress_data:
                preview = job.progress_data.get('preview')
                desc = job.progress_data.get('desc', '')
                html = job.progress_data.get('html', '')
                
                # Always keep preview visible and update its value
                yield None, job_id, gr.update(visible=True, value=preview), desc, html, gr.update(interactive=True), gr.update(interactive=True)

            else:
                # Keep preview visible even when no data
                yield None, job_id, gr.update(visible=True), '', 'Processing...', gr.update(interactive=True), gr.update(interactive=True)
        
        elif job.status == JobStatus.COMPLETED:
            # Don't hide preview on completion
            yield job.result, job_id, gr.update(visible=True), '', '', gr.update(interactive=True), gr.update(interactive=True)
            break
        
        elif job.status == JobStatus.FAILED:
            yield None, job_id, gr.update(visible=True), '', f'Error: {job.error}', gr.update(interactive=True), gr.update(interactive=True)
            break
        
        elif job.status == JobStatus.CANCELLED:
            yield None, job_id, gr.update(visible=True), '', 'Job cancelled', gr.update(interactive=True), gr.update(interactive=True)
            break
        
        # Wait a bit before checking again
        time.sleep(0.5)


# Create the interface using the updated version that supports auto-monitoring
interface = create_interface(
    process_fn=process,
    monitor_fn=monitor_job,
    end_process_fn=end_process,
    update_queue_status_fn=update_queue_status,
    load_lora_file_fn=load_lora_file,
    lora_names=lora_names
)

# Launch the interface
interface.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
