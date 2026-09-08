"""Run locally on a suitable GPU using the same pinned Comfy source and models."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--model-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.argv=[sys.argv[0]];sys.path.insert(0,str(a.comfy_root.resolve()))
from h3_runtime import ensure_models,generate
cfg=json.loads(a.config.read_text());models=ensure_models(json.loads(a.models.read_text()),a.model_cache)
generate(cfg.get('generation',cfg),models,a.output)
