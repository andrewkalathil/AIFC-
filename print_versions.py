"""
print_versions.py - run this in the SAME environment (PyCharm interpreter) that you used for the experiments:

    python print_versions.py

It prints the Python version and the version of every package the pipeline uses, writes requirements_pinned.txt
(use it to replace requirements.txt in the GitHub folder), and prints the sentence for the Implementation section.
"""
import platform
import sys
from importlib import metadata

# (pip name, label used in the paper)
PACKAGES = [
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("scipy", "SciPy"),
    ("scikit-learn", "scikit-learn"),
    ("imbalanced-learn", "imbalanced-learn"),
    ("nibabel", "nibabel"),
    ("tensorflow", "TensorFlow"),
    ("keras", "Keras"),
    ("matplotlib", "Matplotlib"),
    ("lime", "LIME"),
    ("shap", "SHAP (optional cross-check)"),
]

found = {}
print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}\n")
for pip_name, label in PACKAGES:
    try:
        v = metadata.version(pip_name)
        found[pip_name] = (label, v)
        print(f"{label:30s} {v}")
    except metadata.PackageNotFoundError:
        print(f"{label:30s} not installed in this environment")

with open("requirements_pinned.txt", "w") as f:
    for pip_name, (label, v) in found.items():
        f.write(f"{pip_name}=={v}\n")
print("\nWrote requirements_pinned.txt")

v = lambda name: found[name][1] if name in found else "?"
print("\nSentence for the Implementation section (LaTeX):\n")
print(
    f"The pipeline is implemented in Python {platform.python_version()} with NumPy {v('numpy')}, SciPy {v('scipy')}, "
    f"scikit-learn {v('scikit-learn')}, imbalanced-learn {v('imbalanced-learn')} and TensorFlow {v('tensorflow')} "
    f"(Keras {v('keras')}). Shapley values were computed with our own exact implementation, and LIME with the "
    f"\\texttt{{lime}} package (version {v('lime')}). The code is available at \\url{{https://github.com/USERNAME/REPOSITORY}}."
)
