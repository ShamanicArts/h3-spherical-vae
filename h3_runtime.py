"""Pinned Comfy H3 T2V runtime. No research-workspace imports or credentials.

The caller makes the pinned Comfy source importable before calling generate.
Audio latents are retained; this example exports silent ERP video, not audio.
"""
import gc, hashlib, json, math, time, urllib.request
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(16*2**20): h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def artifact_manifest(folder):
    """Hash durable outputs only; progress is a mutable monitoring document."""
    return [{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in sorted(Path(folder).iterdir())
            if p.is_file() and p.name not in ('progress.json','result.json','failed.txt')]


def tensor_sha(t):
    import torch
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()


def ensure_models(manifest,cache):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True); result={}
    for name,entry in manifest.items():
        path=cache/entry['file']
        if not path.exists():
            partial=path.with_suffix('.partial')
            with urllib.request.urlopen(entry['url'],timeout=180) as src,partial.open('wb') as dst:
                while chunk:=src.read(16*2**20):dst.write(chunk)
            if sha(partial)!=entry['sha256']: raise ValueError(f'{name} download hash mismatch')
            partial.replace(path)
        elif sha(path)!=entry['sha256']: raise ValueError(f'{name} cached hash mismatch')
        result[name]=str(path)
    return result


def native_shapes(width,height,frames):
    if any(type(v) is not int or v<=0 for v in (width,height,frames)) or width%64 or height%32:
        raise ValueError('Explicit positive dimensions required; width multiple 64, height multiple 32')
    from comfy_extras.nodes_minimax_h3 import temporal_shape
    actual,vt,at=temporal_shape(frames)
    if actual!=frames: raise ValueError(f'{frames} frames would snap to {actual}; specify a native count')
    return ((1,24,vt,height//16,width//16),(1,32,2,at))


def check_conditioning(cond):
    import torch
    if len(cond)!=1 or cond[0][0].ndim!=3 or cond[0][0].shape[0]!=1 or cond[0][0].shape[-1]!=5120 or not torch.isfinite(cond[0][0]).all():
        raise ValueError('Expected a single finite H3 prompt embedding')
    meta=cond[0][1]
    if any(meta.get(k) is not None for k in ('minimax_keyframes','minimax_refs','minimax_reference_blocks','mask','denoise_mask')):
        raise ValueError('Image/video conditioning is not supported by the final-shift adapter')
    tags=meta.get('minimax_token_tags')
    if tags is None or not torch.all(tags==1): raise ValueError('Text-only token tags required')


def _cpu(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    return value


def export_video(pixels,target,fps):
    import av, numpy as np, torch
    from PIL import Image
    target=Path(target); frames=pixels.shape[2]; hashes=[]; boundaries=[]; snapshots=sorted(set(round(i*(frames-1)/5) for i in range(6)))
    # FFV1 is the metric/reference master. MP4 is a viewing copy.
    with av.open(str(target.with_suffix('.mkv')),'w') as lossless, av.open(str(target.with_suffix('.mp4')),'w') as review:
        a=lossless.add_stream('ffv1',rate=fps); b=review.add_stream('libx264',rate=fps)
        for stream in (a,b):stream.width=pixels.shape[-1];stream.height=pixels.shape[-2]
        a.pix_fmt='bgr0';b.pix_fmt='yuv420p';b.options={'crf':'16','preset':'fast'}
        for index in range(frames):
            rgb=pixels[0,:,index].permute(1,2,0)
            if not torch.isfinite(rgb).all():raise ValueError('Nonfinite decoded pixels')
            rgb=rgb.clamp(0,1).mul(255).round().byte().numpy()
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest()); boundaries.append(float(np.abs(rgb[:,0].astype(np.float32)-rgb[:,-1]).mean()))
            if index in snapshots:Image.fromarray(rgb).save(target.parent/f'{target.name}-frame{index:03d}.png')
            for container,stream in ((lossless,a),(review,b)):
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
        for container,stream in ((lossless,a),(review,b)):
            for packet in stream.encode():container.mux(packet)
    with av.open(str(target.with_suffix('.mkv'))) as container:
        actual=[hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest() for f in container.decode(video=0)]
    if actual!=hashes:raise ValueError('Lossless decoded frames differ')
    return {'frames':frames,'rgb_frame_sha256':hashes,'boundary_mae':sum(boundaries)/frames,'boundary_mae_by_frame':boundaries}


def generate(config,models,output):
    """Full prompt -> joint sampling -> matched control/treatment decodes.

    The output directory must be new: its creation is the single-execution
    claim. Failures are recorded and never silently retried.
    """
    import torch
    from final_shift import build_final_shift_euler
    from circular_decode import decode_circular
    import comfy.sd, comfy.sample, comfy.samplers, comfy.model_management as mm
    from comfy.nested_tensor import NestedTensor
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from safetensors.torch import load_file
    allowed={'prompt','width','height','frames','steps','seed','fps','contexts','final_shift','lora_strength'}
    if set(config)-allowed:raise ValueError('Unknown generation settings')
    width,height,frames=(config[k] for k in ('width','height','frames')); shapes=native_shapes(width,height,frames)
    if type(config['steps']) is not int or not 1<=config['steps']<=1000:raise ValueError('Invalid step count')
    if not isinstance(config['prompt'],str) or not config['prompt'].strip():raise ValueError('Prompt required')
    if type(config['seed']) is not int or not 0<=config['seed']<2**64:raise ValueError('Invalid seed')
    contexts=config.get('contexts',[0,128,384])
    if not contexts or len(set(contexts))!=len(contexts) or any(type(c) is not int or c<0 or c%16 or c>width for c in contexts):raise ValueError('Invalid decoder contexts')
    fps=config.get('fps',24)
    if type(fps) is not int or fps!=24:raise ValueError('This native H3 example uses 24 fps')
    strength=config.get('lora_strength',1.0)
    if not isinstance(strength,(float,int)) or not math.isfinite(strength) or strength!=1:raise ValueError('Validated recipe requires LoRA 1.0')
    output=Path(output);output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    def progress(stage,**kw):
        row={'stage':stage,'elapsed_seconds':time.monotonic()-started,**kw}; atomic_json(output/'progress.json',row);print(json.dumps(row),flush=True)
    try:
        with torch.inference_mode():
            progress('encoding')
            clip=comfy.sd.load_clip(ckpt_paths=[models['text_encoder']],clip_type=comfy.sd.CLIPType.MINIMAX,model_options={'dtype':torch.bfloat16})
            tokens=clip.tokenize(config['prompt'],images=[]);cond=_cpu(clip.encode_from_tokens_scheduled(tokens));check_conditioning(cond)
            torch.save({'conditioning':cond,'prompt':config['prompt']},output/'conditioning.pt')
            del clip,tokens;mm.unload_all_models();gc.collect();mm.soft_empty_cache()
            progress('loading-diffusion')
            model=comfy.sd.load_diffusion_model(models['diffusion'],model_options={'dtype':torch.bfloat16})
            weights=comfy.utils.load_torch_file(models['lora'],safe_load=True)
            model,_=comfy.sd.load_lora_for_models(model,None,weights,strength,0.0);del weights
            latent=NestedTensor([torch.zeros(s,dtype=torch.float32) for s in shapes]);noise=comfy.sample.prepare_noise(latent,config['seed'])
            sigmas=comfy.samplers.calculate_sigmas(model.get_model_object('model_sampling'),'simple',config['steps']).cpu()
            guider=comfy.samplers.CFGGuider(model);guider.inner_set_conds({'positive':cond})
            receipt=[]; captured={}
            def observe(x,canonical,corrected,control,result):
                captured.update(control=control.cpu().clone(),result=result.cpu().clone())
                torch.save({'state':x.cpu(),'canonical_clean':canonical.cpu(),'corrected_clean':corrected.cpu(),'control':control.cpu(),'result':result.cpu(),'shapes':shapes,'sigmas':sigmas},output/'final-audit.pt')
            function=build_final_shift_euler(shapes,enabled=config.get('final_shift',True),receipt=receipt,final_observer=observe)
            tick=time.monotonic();samples=guider.sample(noise,latent,comfy.samplers.KSAMPLER(function),sigmas,denoise_mask=None,callback=lambda i,*_:progress('sampling',step=i+1,steps=config['steps']),disable_pbar=True,seed=config['seed']);torch.cuda.synchronize();sampling_seconds=time.monotonic()-tick
            streams=[s.float().cpu() for s in samples.unbind()];count=math.prod(shapes[0][1:])
            control=[captured['control'][...,:count].reshape(shapes[0]),captured['control'][...,count:].reshape(shapes[1])*.25]
            if not torch.equal(streams[0],captured['result'][...,:count].reshape(shapes[0])) or not torch.equal(streams[1],captured['result'][...,count:].reshape(shapes[1])*.25):raise ValueError('Comfy output normalization differs from audited H3 contract')
            if not torch.equal(streams[1],control[1]) or not torch.equal(streams[0][...,1:-1],control[0][...,1:-1]):raise ValueError('Correction escaped seam columns/audio')
            torch.save({'control':control,'treatment':streams,'config':config,'sigmas':sigmas},output/'latents.pt')
            del guider,model,samples,noise,latent,captured,cond;mm.unload_all_models();gc.collect();mm.soft_empty_cache()
            vae=MiniMaxH3VideoVAE().eval().to(torch.float16);vae.load_state_dict(load_file(models['vae']),strict=True);vae.to('cuda')
            decodes=[];opposite=[]
            for context in contexts:
                reference=None
                for label,video in [('control',control[0]),('treatment',streams[0])]:
                    progress('decoding',arm=label,context=context)
                    buffer=torch.empty((1,3,frames,height,width+2*context),device='cpu',dtype=torch.float32)
                    pixels=decode_circular(vae,video.to('cuda',torch.float16),context_pixels=context,output_buffer=buffer)
                    if tuple(pixels.shape)!=(1,3,frames,height,width):raise ValueError('Decoded native shape changed')
                    central=pixels[...,width//4:3*width//4].clamp(0,1).mul(255).round().byte()
                    if label=='control':reference=central.clone()
                    else:
                        delta=(central.to(torch.int16)-reference.to(torch.int16)).abs()
                        opposite.append({'context':context,'all_frames':frames,'middle_half_max_rgb_difference':int(delta.max()),'middle_half_changed_channels':int(torch.count_nonzero(delta))})
                    record=export_video(pixels,output/f'{label}-context{context}',fps);record.update(arm=label,context=context);decodes.append(record)
                    del pixels,buffer,central
                del reference
            result={'config':config,'shapes':shapes,'sampling_seconds':sampling_seconds,'total_seconds':time.monotonic()-started,'evaluations':sum(r['model_evaluations'] for r in receipt),'steps':receipt,'decodes':decodes,'opposite_checks':opposite,'exterior_latent_and_audio_exact':True,'model_sha256':{k:sha(v) for k,v in models.items()},'artifacts':[]}
            # Hash artifacts before declaring success; no credentials or private paths.
            result['artifacts']=artifact_manifest(output)
            atomic_json(output/'result.json',result);progress('completed');return result
    except BaseException:
        import traceback
        (output/'failed.txt').write_text(traceback.format_exc());raise
