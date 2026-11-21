"""Load the project's existing trusted model files using portable paths.

Pickle/joblib files can execute code. Only load model files you trust.
The legacy directory remains supported so existing repository files need
not be moved or retrained as part of this path-only fix.
"""
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_FILENAMES = ('vectorizer.pkl', 'label_encoder.pkl', 'knn_model.pkl')


def load_models(model_dir=None):
    """Return vectorizer, label encoder and classifier from one complete directory.

    An explicit directory is used exactly as supplied, never silently replaced.
    Otherwise prefer models/, then the repository's legacy checkpoint directory.
    Relative explicit paths are resolved against the repository, not the CWD.
    """
    if model_dir is not None:
        directory = Path(model_dir)
        if not directory.is_absolute():
            directory = ROOT / directory
        candidates = [directory]
    else:
        candidates = [ROOT / 'models', ROOT / '.ipynb_checkpoints']
    for directory in candidates:
        paths = [directory / name for name in MODEL_FILENAMES]
        if all(path.is_file() for path in paths):
            return tuple(joblib.load(path) for path in paths)
    searched = ', '.join(str(directory) for directory in candidates)
    raise FileNotFoundError(
        'Could not find a complete model set. Expected vectorizer.pkl, '
        'label_encoder.pkl and knn_model.pkl together in: ' + searched
    )
