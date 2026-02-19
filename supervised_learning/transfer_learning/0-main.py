#!/usr/bin/env python3
"""Runner to evaluate a saved CIFAR-10 model.

This script loads `preprocess_data` from the training script and
evaluates the saved `cifar10.h5` model on the CIFAR-10 test set.
"""
from pathlib import Path
from tensorflow import keras as K


# Dynamically load preprocessing function from the training script.
import runpy
module_globals = runpy.run_path('supervised_learning/transfer_learning/0-transfer.py')
preprocess_data = module_globals.get('preprocess_data')
build_model = module_globals.get('build_model')
if preprocess_data is None:
    raise ImportError('preprocess_data not found in 0-transfer.py')


def main():
    """Load data, preprocess, load model and evaluate."""
    _, (X, Y) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(X, Y)

    # Resolve model path relative to this file
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / 'cifar10.h5'
    if not model_path.exists():
        alt = Path('supervised_learning/transfer_learning') / 'cifar10.h5'
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path} or {alt}."
            )

    try:
        model = K.models.load_model(str(model_path))
    except Exception:
        # Fallback: build the model architecture and load weights from
        # the HDF5 file (this works when the saved model's config
        # cannot be deserialized by the current Keras version).
        if build_model is None:
            raise RuntimeError(
                'Could not load model and build_model not available from 0-transfer.py'
            )
        model = build_model()
        # Load weights from the HDF5 file created by model.save(..., save_format='h5')
        model.load_weights(str(model_path))
        model.compile(optimizer=K.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.evaluate(X_p, Y_p, batch_size=128, verbose=1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

from tensorflow import keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)