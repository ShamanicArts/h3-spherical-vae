"""H3 masked perspective-video sampling, extracted from the prompt ablation.

The service supplies fresh text-only conditioning and handles compositing.
"""
from pathlib import Path
import hashlib
import json
import re
import time
import urllib.request

MODEL_REVISION = '4cc1d817b6184899b41293954329f576cb5ae86b'
MODEL_FILE = 'diffusion_models/minimax_h3_fl2va_bf16.safetensors'
MODEL_SHA256 = '907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182'
MODEL_BYTES = 66280487368
MODEL_URL = f'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{MODEL_REVISION}/{MODEL_FILE}'
VAE_SHA256 = '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'
LORA_SHA256 = 'a1e3ebd2b79be92d5e970c6f516519f8bd7f8e4271a71e7a7d7c8e938e157cf7'


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class H3PolarRepair:
    def __init__(self, vae_path, diffusion_path, output_dir,
                 lora_path=None, lora_strength=0.0, verify_weights=True):
        import torch
        from safetensors.torch import load_file
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        if verify_weights and digest(vae_path) != VAE_SHA256:
            raise ValueError('VAE SHA mismatch')
        self.diffusion_path = str(diffusion_path)
        self.lora_path, self.lora_strength = lora_path, float(lora_strength)
        if not 0 <= self.lora_strength <= 1:
            raise ValueError('Probe LoRA strength must be in [0,1]')
        if self.lora_strength and not self.lora_path:
            raise ValueError('Nonzero LoRA strength requires a path')
        if self.lora_path and verify_weights and digest(self.lora_path) != LORA_SHA256:
            raise ValueError('Reviewed LoRA SHA mismatch')
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.vae = MiniMaxH3VideoVAE().eval().to(dtype=torch.float16)
        self.vae.load_state_dict(load_file(str(vae_path)), strict=True)
        self.model = None
        self.verify_weights = verify_weights

    def _model(self):
        if self.model is None:
            import comfy.sd
            import torch
            import comfy.utils
            if self.verify_weights:
                if Path(self.diffusion_path).stat().st_size != MODEL_BYTES or digest(self.diffusion_path) != MODEL_SHA256:
                    raise ValueError('Diffusion SHA/size mismatch')
            self.model = comfy.sd.load_diffusion_model(self.diffusion_path, model_options={'dtype': torch.bfloat16})
            if self.model is None:
                raise RuntimeError('Comfy failed to recognize native H3 diffusion model')
            if self.lora_strength:
                weights = comfy.utils.load_torch_file(str(self.lora_path), safe_load=True)
                self.model, _ = comfy.sd.load_lora_for_models(
                    self.model, None, weights, self.lora_strength, 0.0)
        return self.model

    def run(self, case_id, frames_uint8, conditioning, audio_latent, denoise_mask,
            strength=1.0, steps=32, seed=2026090911, zero_denoise=False, structure_strength=0.0):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from comfy.nested_tensor import NestedTensor
        import comfy.model_management as mm
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', case_id):
            raise ValueError('case_id must be a safe basename')
        frames = frames_uint8.detach().cpu().numpy() if isinstance(frames_uint8, torch.Tensor) else np.asarray(frames_uint8)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError('Expected uint8 [T,H,W,3] RGB frames')
        length, height, width, _ = frames.shape
        if length not in (124,) or width % 32 or height % 32:
            raise ValueError('Repair currently supports exactly 124 frames, dimensions divisible by 32')
        if width > 1024 or height > 1024 or min(width, height) < 256:
            raise ValueError('Bounded perspective dimensions must be256..1024')
        if not zero_denoise and (not 0 < strength <= 1.0 or not 1 <= steps <= 40):
            raise ValueError('Bounded probe requires sigma_start in(0,1], steps1..40')
        mask = denoise_mask.detach().float().cpu() if isinstance(denoise_mask, torch.Tensor) else torch.as_tensor(np.asarray(denoise_mask), dtype=torch.float32)
        if tuple(mask.shape) != (height, width) or not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError('Mask must be finite H×W in[0,1],1 meansrepair')
        if not torch.any(mask > 0) or not torch.any(mask == 0):
            raise ValueError('Mask must contain both editable and exactlypreserved pixels')
        started = time.monotonic()
        timing = {}
        target = self.output / case_id
        if target.with_suffix('.json').exists():
            raise FileExistsError('Completed case exists; controller must reuse receipt, not regenerate')
        mm.unload_all_models()
        self.vae.to('cuda')
        pixels = torch.from_numpy(frames.copy()).float().permute(3, 0, 1, 2).unsqueeze(0).div_(127.5).sub_(1)
        with torch.inference_mode():
            torch.cuda.synchronize()
            t = time.monotonic()
            video_z = self.vae.encode(pixels.to(dtype=torch.float16), device='cuda').float().cpu()
            torch.cuda.synchronize()
            timing['encode_seconds'] = time.monotonic() - t
            self.vae.cpu()
            mm.soft_empty_cache()
            latent_t = 2 + (length - 5) // 17 * 5
            if tuple(video_z.shape) != (1, 24, latent_t, height // 16, width // 16):
                raise ValueError(f'Unexpected VAE latent shape {video_z.shape}')
            audio_t = round(length / 24 * 40)
            audio_z = audio_latent.detach().float().cpu()[..., :audio_t].contiguous()
            if tuple(audio_z.shape) != (1, 32, 2, audio_t):
                raise ValueError(f'Expected source audio slice [1,32,2,{audio_t}]')
            latent = NestedTensor([video_z, audio_z])
            # Align repair mask to H3's 32pixel DiT cells to avoid accidental
            # partialcell temporal labels; caller retains fullpixel blend mask.
            grid = F.max_pool2d(mask[None, None], kernel_size=32, stride=32)
            grid = grid.repeat_interleave(2, -2).repeat_interleave(2, -1)
            video_mask = grid.unsqueeze(2).expand(1, 24, latent_t, height // 16, width // 16).contiguous()
            noise_mask = NestedTensor([video_mask, torch.zeros_like(audio_z)])
            sigmas = torch.linspace(float(strength), 0.0, steps + 1) if not zero_denoise else torch.zeros(1)
            callbacks = []
            guidance_records = []
            assert 0 <= structure_strength <= .5
            if zero_denoise:
                result_z = video_z
                audio_change = 0.0
                timing['sampling_seconds'] = 0.0
            else:
                import comfy.sample
                import comfy.samplers
                # BasicGuider equivalent, avoiding UI node imports.
                if structure_strength:
                    raise ValueError('Structure guidance is not exposed by this service')
                guider = comfy.samplers.CFGGuider(self._model())
                guider.inner_set_conds({'positive': conditioning})
                noise = comfy.sample.prepare_noise(latent, seed)
                noise_sha256 = [hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for x in noise.unbind()]
                def callback(step, x0, x, total_steps):
                    callbacks.append({'step': int(step), 'total_steps': int(total_steps),
                                      'elapsed_seconds': time.monotonic() - t})
                    print(json.dumps({'case': case_id, 'step': int(step) + 1, 'steps': total_steps}), flush=True)
                torch.cuda.synchronize()
                t = time.monotonic()
                sampled = guider.sample(noise, latent, comfy.samplers.ksampler('res_multistep'),
                    sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=True, seed=seed)
                torch.cuda.synchronize()
                timing['sampling_seconds'] = time.monotonic() - t
                result_z, result_audio = [x.float().cpu() for x in sampled.unbind()]
                audio_change = float((result_audio - audio_z).abs().max())
                if structure_strength:
                    assert guidance_records and max(r['mean_abs_correction'] for r in guidance_records)>0
                    assert guidance_records[-1]['mean_abs_correction']==0, 'Late guidance release missing'
                if not torch.isfinite(result_z).all():
                    raise RuntimeError('Nonfinite sampled video latent')
                mm.unload_all_models()
                mm.soft_empty_cache()
            self.vae.to('cuda')
            t = time.monotonic()
            decoded = self.vae.decode(result_z.to('cuda', dtype=torch.float16)).float().cpu()
            torch.cuda.synchronize()
            timing['decode_seconds'] = time.monotonic() - t
            self.vae.cpu()
            mm.soft_empty_cache()
        if tuple(decoded.shape) != (1, 3, length, height, width) or not torch.isfinite(decoded).all():
            raise RuntimeError(f'Invalid decoded result {decoded.shape}')
        out = decoded[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().byte().numpy()
        np.save(target.with_suffix('.npy'), out, allow_pickle=False)
        for i, rgb in enumerate(out):
            if i not in (0, len(out)//2, len(out)-1):continue
            Image.fromarray(rgb).save(self.output / f'{case_id}-frame{i:03d}.png')
        torch.save({'video': result_z, 'input_video': video_z, 'mask': video_mask,
                    'sigmas': sigmas, 'seed': seed}, target.with_suffix('.pt'))
        report = {'case_id': case_id, 'kind': 'vae_roundtrip_control' if zero_denoise else 'true_masked_diffusion',
            'shape': list(frames.shape), 'latent_shape': list(video_z.shape),
            'model_sha256': MODEL_SHA256, 'vae_sha256': VAE_SHA256,
            'lora_strength': self.lora_strength, 'seed': seed, 'diffusion_dtype': 'torch.bfloat16', 'vae_dtype': 'torch.float16',
            'steps': 0 if zero_denoise else steps, 'sampler': 'res_multistep',
            'schedule': 'explicit linear sigma_start→0; NOT BasicScheduler denoisefraction',
            'structure_strength': structure_strength, 'structure_guidance_calls': guidance_records,
            'sigmas': sigmas.tolist(), 'audio_mask': 'allzero; sourceaudio latentheldfixed',
            'audio_max_abs_change': audio_change, 'sampling_callbacks': callbacks,
            'conditioning_caveat': 'Caller supplies freshly encoded, explicitly recorded repair text; no image tokens.',
            'short_clip_caveat': '5/22/39frames below reported trainingrange124..362' if length < 124 else None,
            'initial_noise_sha256': noise_sha256, 'input_video_latent_sha256': hashlib.sha256(video_z.contiguous().numpy().tobytes()).hexdigest(),
            'mask_grid_pixels': 32, 'timing': timing, 'total_seconds': time.monotonic() - started,
            'output_npy': str(target.with_suffix('.npy')), 'output_sha256': digest(target.with_suffix('.npy')),
            'outside_mask_latent_max_change': float(((result_z - video_z) * (video_mask == 0)).abs().max()),
            'note': 'Pixels outside latentmask maychange through VAE reconstruction. Backprojection must retain original ERP outside pixelcomposite mask.'}
        target.with_suffix('.json').write_text(json.dumps(report, indent=2))
        return report
