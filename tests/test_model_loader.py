"""Path tests with mocked deserialization. No repository pickle is executed."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import model_loader


class ModelPathTests(unittest.TestCase):
    def create_files(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name in model_loader.MODEL_FILENAMES:
            (directory / name).touch()

    def test_legacy_files_work_from_other_working_directory(self):
        with TemporaryDirectory() as tmp, patch.object(model_loader.joblib, 'load', side_effect=lambda p: p) as load:
            old = Path.cwd()
            try:
                os.chdir(tmp)
                paths = model_loader.load_models()
            finally:
                os.chdir(old)
        self.assertEqual(load.call_count, 3)
        self.assertTrue(all(p.parent == model_loader.ROOT / '.ipynb_checkpoints' for p in paths))

    def test_prefers_complete_models_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_files(root / 'models')
            self.create_files(root / '.ipynb_checkpoints')
            with patch.object(model_loader, 'ROOT', root), patch.object(model_loader.joblib, 'load', side_effect=lambda p: p):
                paths = model_loader.load_models()
            self.assertTrue(all(p.parent == root / 'models' for p in paths))

    def test_incomplete_set_does_not_deserialize_or_mix_directories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models').mkdir()
            (root / 'models' / 'vectorizer.pkl').touch()
            with patch.object(model_loader, 'ROOT', root), patch.object(model_loader.joblib, 'load') as load:
                with self.assertRaisesRegex(FileNotFoundError, 'complete model set'):
                    model_loader.load_models()
                load.assert_not_called()

    def test_explicit_relative_directory_uses_repository_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_files(root / 'custom')
            with patch.object(model_loader, 'ROOT', root), patch.object(model_loader.joblib, 'load', side_effect=lambda p: p):
                paths = model_loader.load_models('custom')
            self.assertTrue(all(p.parent == root / 'custom' for p in paths))

    def test_missing_explicit_directory_does_not_fall_back(self):
        with patch.object(model_loader.joblib, 'load') as load:
            with self.assertRaises(FileNotFoundError):
                model_loader.load_models('does-not-exist')
            load.assert_not_called()

if __name__ == '__main__':
    unittest.main()
