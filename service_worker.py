"""One isolated GPU stage. Process exit releases model references and CUDA RAM."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','repair']);p.add_argument('job',type=Path)
    args=p.parse_args();job=json.loads(args.job.read_text())
    # Parse service arguments before importing Comfy's process-level CLI parser.
    import sys
    sys.argv=[sys.argv[0]]
    from service_schema import SphericalRequest
    request=SphericalRequest(**job['input'])
    if args.stage=='generate':
        from h3_runtime import generate
        generate(request.generation_config(),job['models'],job['output'])
    else:
        from polar_pipeline import repair
        repair(job['source'],request,job['models'],job['output'])


if __name__=='__main__':main()
