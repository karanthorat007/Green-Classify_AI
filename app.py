import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)

if sys.version_info >= (3, 12):
    _fail(
        "Python 3.12+ is not supported by TensorFlow in this project. "
        "Use Python 3.11 and the .venv interpreter:\n"
        "  .\\.venv\\Scripts\\python.exe app.py"
    )

if ".venv" not in str(Path(sys.executable).resolve()):
    _fail(
        "You're not running the project virtual environment. "
        "Use:\n  .\\.venv\\Scripts\\Activate.ps1\n  python app.py"
    )

try:
    import tensorflow as tf  # noqa: F401
except Exception as exc:  # pragma: no cover - startup guard
    _fail(
        "TensorFlow is not installed for this interpreter. "
        "Activate .venv and install requirements:\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )

from keras import __version__ as keras_version
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
UPLOAD_DIR = os.path.join(app.root_path, UPLOAD_FOLDER)
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not keras_version.startswith("3."):
    _fail(
        f"Keras {keras_version} is installed, but Keras 3 is required to load "
        "this model. Reinstall requirements in .venv."
    )

model = load_model("vegetable_classifier_model.h5", compile=False)

classes = [
    'Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli',
    'Cabbage','Capsicum','Carrot','Cauliflower','Cucumber',
    'Papaya','Potato','Pumpkin','Radish','Tomato'
]

HINDI_NAMES = {
    "Bean": "सेम",
    "Bitter_Gourd": "करेला",
    "Bottle_Gourd": "लौकी",
    "Brinjal": "बैंगन",
    "Broccoli": "ब्रोकली",
    "Cabbage": "पत्ता गोभी",
    "Capsicum": "शिमला मिर्च",
    "Carrot": "गाजर",
    "Cauliflower": "फूलगोभी",
    "Cucumber": "खीरा",
    "Papaya": "पपीता",
    "Potato": "आलू",
    "Pumpkin": "कद्दू",
    "Radish": "मूली",
    "Tomato": "टमाटर",
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_en = None
    prediction_hi = None
    confidence = None
    is_vegetable = True
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "index.html",
                prediction=None,
                image_path=None,
                confidence=None,
                is_vegetable=True
            )

        file = request.files["file"]
        filename = secure_filename(file.filename)
        if file and filename:
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)
            image_path = os.path.join(UPLOAD_FOLDER, filename)

            img = load_img(save_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            confidence = float(np.max(pred))
            prediction = classes[np.argmax(pred)]
            is_vegetable = confidence >= 0.5
            if not is_vegetable:
                prediction = None
            else:
                prediction_en = prediction.replace("_", " ").title()
                prediction_hi = HINDI_NAMES.get(prediction, prediction_en)

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_en=prediction_en,
        prediction_hi=prediction_hi,
        image_path=image_path,
        confidence=confidence,
        is_vegetable=is_vegetable
    )

if __name__ == "__main__":
    app.run(debug=True)
