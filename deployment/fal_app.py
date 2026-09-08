"""Deploy the built wheel and pinned Comfy source, with a single run identity."""
import json
from pathlib import Path
import fal
from pydantic import BaseModel

class Request(BaseModel):
    run_id: str

class H3Spherical(fal.App):
    app_name='h3-spherical-validation'
    app_auth='private'
    machine_type='GPU-H200'
    num_gpus=1
    min_concurrency=0
    max_concurrency=1
    concurrency_buffer=0
    concurrency_buffer_perc=0
    max_multiplexing=1
    keep_alive=60
    startup_timeout=3600
    request_timeout=3600
    skip_retry_conditions=['timeout','server_error','connection_error']
    secrets=[]
    requirements=['torch==2.13.0', 'numpy==2.5.3', 'safetensors==0.8.0', 'einops==0.8.2', 'psutil==7.2.2', 'scipy==1.18.1', 'av==18.1.0', 'Pillow==12.3.0', 'tqdm==4.70.0', 'comfy-kitchen==0.2.31', 'comfy-aimdo==0.4.15', 'pyyaml==6.0.3', 'torchsde', 'torchaudio', 'torchvision', 'transformers==5.16.1', 'tokenizers==0.23.2', 'sentencepiece==0.2.2', 'aiohttp', 'yarl', 'filelock', 'requests', 'simpleeval', 'blake3']
    app_files=['runtime.whl','comfy-source.tar.gz','models.json','config.json']
    app_files_context_dir=str(Path(__file__).resolve().parent)
    def setup(self):
        import hashlib,sys,tarfile,threading,os,zipfile
        here=Path.cwd()
        self.config=json.loads((here/'config.json').read_text())
        self.guard=threading.Timer(self.config['worker_guard_seconds'],lambda:os._exit(124));self.guard.daemon=True;self.guard.start()
        archive=here/'comfy-source.tar.gz'
        if hashlib.sha256(archive.read_bytes()).hexdigest()!='2fb9bb75a0c2c377f408fea2f06c3b68ccf33505ed3c1332d830e4d5cb44bc82':raise ValueError('Comfy source hash mismatch')
        vendor=Path('/tmp/h3-spherical-vendor');vendor.mkdir(exist_ok=False)
        with tarfile.open(archive) as t:t.extractall(vendor,filter='data')
        package=Path('/tmp/h3-spherical-package');package.mkdir(exist_ok=False)
        with zipfile.ZipFile(here/'runtime.whl') as z:z.extractall(package)
        sys.path[:0]=[str(package),str(next(vendor.iterdir()))];sys.argv=[sys.argv[0]]
        import h3_runtime,final_shift,circular_decode,spherical_context
        self.runtime=h3_runtime
        self.source_hashes={m.__name__:h3_runtime.sha(m.__file__) for m in (h3_runtime,final_shift,circular_decode,spherical_context)}
        self.models=json.loads((here/'models.json').read_text())
        self.receipt={'wheel_sha256':h3_runtime.sha(here/'runtime.whl'),'source_sha256':self.source_hashes,'comfy_commit':'12d5279438bfefc058a269eae805ceab6047777f'}
        print('H3 spherical wheel loaded',flush=True)
    @fal.endpoint('/')
    def run(self,request:Request)->dict:
        if request.run_id!=self.config['run_id']:raise ValueError('Unknown run ID')
        root=Path(self.config['output_root']);root.mkdir(parents=True,exist_ok=True)
        # Claim before expensive loading; prevents duplicates on another worker.
        with (root/'intent.json').open('x') as f:json.dump({'run_id':request.run_id,**self.receipt},f)
        self.runtime.atomic_json(root/'deployment.json',self.receipt)
        models=self.runtime.ensure_models(self.models,self.config['model_cache'])
        return self.runtime.generate(self.config['generation'],models,root/'generation')
