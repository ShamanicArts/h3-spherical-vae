import tempfile,unittest
from pathlib import Path
from h3_runtime import artifact_manifest,sha
class ArtifactManifestTests(unittest.TestCase):
    def test_mutable_status_is_not_a_hashed_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'frame.png').write_bytes(b'lossless-fixture')
            for name in ('progress.json','result.json','failed.txt'):(root/name).write_text('status')
            a=artifact_manifest(root);(root/'progress.json').write_text('completed')
            self.assertEqual(a,artifact_manifest(root));self.assertEqual(a,[{'name':'frame.png','bytes':16,'sha256':sha(root/'frame.png')}])
if __name__=='__main__':unittest.main()
