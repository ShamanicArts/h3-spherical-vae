"""Process-level tests; fal is optional for the base geometry package."""
import asyncio
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

HAS_FAL=importlib.util.find_spec('fal') is not None


@unittest.skipUnless(HAS_FAL,'Install fal to test the service process wrapper')
class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_releases_the_actual_child_process(self):
        import os
        import sys
        spec=importlib.util.spec_from_file_location('service_app_test',Path(__file__).resolve().parents[1]/'deployment/service_app.py')
        app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'service_worker.py').write_text(
                "import os,time\nfrom pathlib import Path\n"
                f"Path({str(root/'pid')!r}).write_text(str(os.getpid()))\n"
                "time.sleep(60)\n")
            fake=SimpleNamespace(env={**os.environ,'PYTHONPATH':str(root)})
            task=asyncio.create_task(app.H3SphericalService.stage(fake,'repair',{},root))
            for _ in range(100):
                if (root/'pid').exists():break
                await asyncio.sleep(.05)
            self.assertTrue((root/'pid').exists())
            pid=int((root/'pid').read_text());task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=='__main__':unittest.main()
