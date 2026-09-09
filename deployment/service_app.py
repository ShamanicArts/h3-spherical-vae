"""Repeatable fal endpoint built only from the clean wheel and pinned sources."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import tarfile
import tempfile
import time
import zipfile
import fal
import fastapi
from fal.exceptions import RequestCancelledException
from fal.toolkit import File, Video, download_file
from pydantic import BaseModel
# fal loads the entry file before mounting app_files. Resolve the sibling schema
# locally; local_python_modules below ships it for remote deserialization.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_schema import SphericalRequest


class SphericalOutput(BaseModel):
    video: Video
    unrepaired_video: Video
    canonical_video: Video | None = None
    receipt: File
    request_id: str
    elapsed_seconds: float


class H3SphericalService(fal.App):
    app_name='h3-spherical'
    app_auth='private'
    local_python_modules=['service_schema']
    machine_type='GPU-H200'
    num_gpus=1
    min_concurrency=0
    max_concurrency=1
    concurrency_buffer=0
    concurrency_buffer_perc=0
    max_multiplexing=1
    keep_alive=30
    startup_timeout=1800
    request_timeout=3600
    skip_retry_conditions=['timeout','server_error','connection_error']
    secrets=[]
    requirements=['torch==2.13.0', 'numpy==2.5.3', 'safetensors==0.8.0', 'einops==0.8.2', 'psutil==7.2.2', 'scipy==1.18.1', 'av==18.1.0', 'Pillow==12.3.0', 'tqdm==4.70.0', 'comfy-kitchen==0.2.31', 'comfy-aimdo==0.4.15', 'pyyaml==6.0.3', 'torchsde', 'torchaudio', 'torchvision', 'transformers==5.16.1', 'tokenizers==0.23.2', 'sentencepiece==0.2.2', 'aiohttp', 'yarl', 'filelock', 'requests', 'simpleeval', 'blake3', 'imageio-ffmpeg==0.6.0']
    app_files=['runtime.whl','comfy-source.tar.gz','models.json','service_schema.py']
    app_files_context_dir=str(Path(__file__).resolve().parent)

    def setup(self):
        self._tasks={}
        self.base=Path(tempfile.mkdtemp(prefix='h3-spherical-'))
        mounted=Path.cwd();vendor=self.base/'vendor';package=self.base/'package'
        vendor.mkdir();package.mkdir()
        archive=mounted/'comfy-source.tar.gz'
        if hashlib.sha256(archive.read_bytes()).hexdigest()!='2fb9bb75a0c2c377f408fea2f06c3b68ccf33505ed3c1332d830e4d5cb44bc82':
            raise ValueError('Pinned Comfy source hash mismatch')
        with tarfile.open(archive) as t:t.extractall(vendor,filter='data')
        with zipfile.ZipFile(mounted/'runtime.whl') as z:z.extractall(package)
        self.env={**os.environ,'PYTHONPATH':os.pathsep.join([str(package),str(next(vendor.iterdir()))]),
                  'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        sys.path.insert(0,str(package))
        from h3_runtime import ensure_models,sha
        from filelock import FileLock
        # Cache on fal persistent storage; no training data or credentials bundled.
        with FileLock('/data/h3-spherical-models.lock'):
            self.models=ensure_models(json.loads((mounted/'models.json').read_text()),'/data/h3-vae/models')
        self.provenance={'wheel_sha256':sha(mounted/'runtime.whl'),
            'models':json.loads((mounted/'models.json').read_text()),
            'comfy_commit':'12d5279438bfefc058a269eae805ceab6047777f'}
        print('H3 spherical service ready',flush=True)

    async def stage(self,stage,job,root):
        path=root/f'{stage}-job.json';path.write_text(json.dumps(job))
        process=await asyncio.create_subprocess_exec(sys.executable,'-m','service_worker',stage,str(path),
                    env=self.env,cwd=root,start_new_session=True)
        try:
            code=await process.wait()
            if code:raise RuntimeError(f'{stage} failed (exit {code}); inspect request logs')
        finally:
            if process.returncode is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:await asyncio.wait_for(process.wait(),10)
                except asyncio.TimeoutError:
                    os.killpg(process.pid,signal.SIGKILL);await process.wait()

    async def execute(self,input,ident):
        from h3_runtime import sha,atomic_json
        started=time.monotonic()
        root=Path('/data/h3-spherical/requests')/ident
        # Durable request claim across runner replacement. A repeat never generates again.
        root.mkdir(parents=True,exist_ok=True)
        if (root/'response.json').exists():
            previous=json.loads((root/'intent.json').read_text())
            if previous['input']!=input.model_dump():
                raise ValueError('Request ID belongs to different inputs')
            return SphericalOutput(**json.loads((root/'response.json').read_text()))
        with (root/'intent.json').open('x') as f:
            json.dump({'input':input.model_dump(),'at':time.time()},f)
        canonical=None
        try:
            if input.video_url:
                source=await asyncio.to_thread(download_file,input.video_url,root/'input',filesize_limit=500)
                # Validate before any model encoding or sampling, including no-op requests.
                from polar_pipeline import read_video,save_video
                frames=await asyncio.to_thread(read_video,source)
                if frames.shape[1:3]!=(input.height,input.width):raise ValueError('Input dimensions differ from requested dimensions')
                unrepaired=root/'unrepaired.mp4'
                await asyncio.to_thread(save_video,unrepaired,frames)
            else:
                await self.stage('generate',{'input':input.model_dump(),'models':self.models,'output':str(root/'generation')},root)
                context=input.context_pixels if input.circular_decode else 0
                source=root/'generation'/f'treatment-context{context}.mkv'
                unrepaired=source.with_suffix('.mp4')
                canonical=root/'generation'/f'control-context{context}.mp4'
            if input.poles():
                await self.stage('repair',{'input':input.model_dump(),'models':self.models,'source':str(source),'output':str(root/'repair')},root)
                final=root/'repair/repaired.mp4'
            else:final=unrepaired
            record={'request_id':ident,'input':input.model_dump(),'provenance':self.provenance,
                    'source_sha256':sha(source),'output_sha256':sha(final),'unrepaired_sha256':sha(unrepaired),
                    'generation_applied':not bool(input.video_url),
                    'circular_decode_applied':not bool(input.video_url) and input.circular_decode,
                    'final_shift_applied':not bool(input.video_url) and input.final_shift,
                    'audio':'Silent outputs; audio generation/preservation not implemented.',
                    'elapsed_seconds':time.monotonic()-started}
            for stage in ('generation','repair'):
                p=root/stage/'result.json'
                if p.exists():record[stage]=json.loads(p.read_text())
            atomic_json(root/'receipt.json',record)
            response=SphericalOutput(video=await asyncio.to_thread(Video.from_path,final,content_type='video/mp4'),
                unrepaired_video=await asyncio.to_thread(Video.from_path,unrepaired,content_type='video/mp4'),
                canonical_video=await asyncio.to_thread(Video.from_path,canonical,content_type='video/mp4') if canonical else None,
                receipt=await asyncio.to_thread(File.from_path,root/'receipt.json',content_type='application/json'),request_id=ident,
                elapsed_seconds=record['elapsed_seconds'])
            atomic_json(root/'response.json',response.model_dump(mode='json'))
            return response
        except BaseException:
            import traceback
            (root/'failed.txt').write_text(traceback.format_exc());raise

    @fal.endpoint('/')
    async def run(self,input:SphericalRequest,x_fal_request_id:str|None=fastapi.Header(None))->SphericalOutput:
        if not x_fal_request_id or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',x_fal_request_id):
            raise ValueError('Use the fal queue API; a valid request ID is required')
        if x_fal_request_id in self._tasks:raise ValueError('Request is already running')
        task=asyncio.create_task(self.execute(input,x_fal_request_id));self._tasks[x_fal_request_id]=task
        try:return await asyncio.wait_for(task,3300)
        except asyncio.CancelledError:raise RequestCancelledException('Request cancelled; GPU subprocess stopped')
        finally:self._tasks.pop(x_fal_request_id,None)

    @fal.endpoint('/cancel')
    async def cancel(self,x_fal_request_id:str|None=fastapi.Header(None))->dict:
        task=self._tasks.get(x_fal_request_id)
        if task:task.cancel()
        return {'cancellation_requested':task is not None}
