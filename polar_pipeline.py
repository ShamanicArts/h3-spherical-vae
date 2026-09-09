"""Complete-video pole repair with the prompt-ablation recipe.

No native re-generation follows the composite. The main generation prompt is
never available to the repair encoder. Geometry is confined to projection.
"""
import gc
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from h3_runtime import atomic_json, sha
from polar_geometry import Camera, _inverse_grid


def ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_video(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != 24:
            raise ValueError('Expected exactly 24 fps; no implicit retiming')
        if (stream.width, stream.height) not in ((1024,448),(1536,672),(512,512)):
            raise ValueError('Unsupported dimensions; no implicit resize')
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames)>124:
                raise ValueError('Expected exactly 124 frames; no implicit trim')
        if len(frames)!=124:
            raise ValueError('Expected exactly 124 frames')
        return np.stack(frames)


def small_mask():
    y,x=np.mgrid[:512,:512]
    theta=np.arctan(np.hypot((x+.5)/256-1,(y+.5)/256-1))
    a=np.clip((np.deg2rad(22)-theta)/np.deg2rad(6),0,1)
    # Preserve the historical PNG mask quantization, not an untested float mask.
    return np.rint(a*a*(3-2*a)*255).astype(np.uint8).astype(np.float32)/255


def project(source, pitch, target):
    subprocess.run([ffmpeg(),'-v','error','-nostdin','-threads','1',
        '-filter_threads','1','-i',str(source),'-vf',
        f'v360=input=equirect:output=flat:pitch={pitch}:h_fov=90:v_fov=90:w=512:h=512,fps=24',
        '-an','-c:v','ffv1','-pix_fmt','bgr0','-threads','2',str(target)],check=True)
    return read_video(target)


def composite(original, reference, fixed, pitch):
    """Same residual interpolation as the experiment; zero delta stays exact."""
    from scipy.ndimage import map_coordinates
    camera=Camera(512,512,90,0,pitch)
    gx,gy,support=_inverse_grid(original.shape[1:3],camera)
    coords=np.array([gy[support],gx[support]])
    output=original.copy()
    for i in range(len(original)):
        delta=fixed[i].astype(np.float32)-reference[i]
        values=np.stack([map_coordinates(delta[...,ch],coords,order=1,mode='nearest') for ch in range(3)],-1)
        output[i,support]=np.rint(np.clip(original[i,support].astype(np.float32)+values,0,255)).astype(np.uint8)
    if not np.array_equal(output[:,~support],original[:,~support]):
        raise RuntimeError('Composition changed pixels outside the pole view')
    return output


def save_video(path, frames):
    import av
    lossless=Path(path).suffix=='.mkv'
    with av.open(str(path),'w',options={} if lossless else {'movflags':'+faststart'}) as container:
        stream=container.add_stream('ffv1' if lossless else 'libx264',rate=24)
        stream.width=frames.shape[2];stream.height=frames.shape[1]
        stream.pix_fmt='bgr0' if lossless else 'yuv420p'
        stream.options={'threads':'2'} if lossless else {'crf':'16','preset':'fast','threads':'2'}
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def repair(source, request, models, output):
    import torch
    import comfy.sd
    import comfy.model_management as mm
    from h3_runtime import _cpu, check_conditioning
    from polar_kernel import H3PolarRepair
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original=read_video(source)
    if original.shape[1:3]!=(request.height,request.width):
        raise ValueError('Input dimensions do not match requested dimensions')
    poles=request.poles()
    if not poles:raise ValueError('No pole repair requested')
    alpha=small_mask();conditions={}
    with torch.inference_mode():
        clip=comfy.sd.load_clip(ckpt_paths=[models['text_encoder']],clip_type=comfy.sd.CLIPType.MINIMAX,model_options={'dtype':torch.bfloat16})
        for name,pitch,prompt in poles:
            print(json.dumps({'stage':'repair-text','pole':name}),flush=True)
            tokens=clip.tokenize(prompt,images=[])
            condition=_cpu(clip.encode_from_tokens_scheduled(tokens));check_conditioning(condition)
            torch.save({'conditioning':condition,'prompt':prompt},output/f'{name}-conditioning.pt')
            conditions[name]=condition
        del clip,tokens,condition
        mm.unload_all_models();gc.collect();mm.soft_empty_cache()
        probe=H3PolarRepair(models['vae'],models['diffusion'],output/'latents',verify_weights=False)
        audio=torch.zeros((1,32,2,round(124/24*40)),dtype=torch.float32)
        result=original.copy();reports=[]
        for name,pitch,prompt in poles:
            print(json.dumps({'stage':'repair','pole':name}),flush=True)
            reference=project(source,pitch,output/f'{name}-source.mkv')
            report=probe.run(name,reference,conditions[name],audio,(alpha>0).astype(np.float32),
                             strength=1.,steps=request.repair_steps,seed=request.repair_seed)
            raw=np.load(report.pop('output_npy'))
            fixed=np.rint(np.clip(reference.astype(np.float32)+(raw.astype(np.float32)-reference)*alpha[None,:,:,None],0,255)).astype(np.uint8)
            if not np.array_equal(fixed[:,alpha==0],reference[:,alpha==0]):
                raise RuntimeError('Repair escaped pixel mask')
            result=composite(result,reference,fixed,pitch)
            save_video(output/f'{name}-repaired.mkv',fixed)
            save_video(output/f'{name}-repaired.mp4',fixed)
            report.update(pole=name,pitch=pitch,prompt_text=prompt,
                source_sha256=sha(output/f'{name}-source.mkv'),
                conditioning_sha256=sha(output/f'{name}-conditioning.pt'),
                pixel_exterior_exact=True,mask_core_degrees=16,mask_outer_degrees=22,
                geometry_preprocessing=False,conditioning_images=0)
            atomic_json(output/f'{name}-receipt.json',report);reports.append(report)
    save_video(output/'repaired.mkv',result);save_video(output/'repaired.mp4',result)
    if not np.array_equal(read_video(output/'repaired.mkv'),result):
        raise RuntimeError('Lossless output did not round-trip')
    receipt={'input_sha256':sha(source),'poles':reports,'frames':124,'fps':24,
        'width':request.width,'height':request.height,
        'composition':'original + backproject(soft-masked repair - perspective input)',
        'audio':'Silent output. Audio generation/preservation is not implemented.',
        'output_sha256':sha(output/'repaired.mkv')}
    atomic_json(output/'result.json',receipt)
    return receipt
