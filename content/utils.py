from pathlib import Path
import os, sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def import_polito(
    classes=["H", "IR", "OR"],
    rpms=[523, 937],
    radial_forces=[62.4, 124.8],
    axial_forces=[0, 49],
    measurement_max_len=None,
    bearing_index=3,
    verbose=True,
):
    """
    Import the Polito dataset and return the relevant data as a dictionary.

    Description of the dataset can be found here: https://data.niaid.nih.gov/resources?id=zenodo_13913253
    Article with more details about the dataset can be found here: https://www.mdpi.com/1424-8220/23/1/211


    """

    # Helpers
    class_map = {"H": "Healthy", "IR": "InnerRaceDamage", "OR": "OuterRaceDamage"}
    class_map_inverse = {v: k for k, v in class_map.items()}

    # Glob all dataset .mat files
    mat_files = (Path("data") / "PolitoBearingFaultData").glob("**/*.mat")

    # Filter files based on the specified classes, rpms, radial forces, and axial forces
    mat_files = filter(lambda f: class_map_inverse[f.parent.name] in classes, mat_files)
    mat_files = filter(
        lambda f: int(f.name.split("_")[0].replace("rpm", "")) in rpms, mat_files
    )
    mat_files = filter(
        lambda f: float(f.name.split("_")[1].replace("kN", "")) in radial_forces,
        mat_files,
    )
    mat_files = filter(
        lambda f: float(f.name.split("_")[2].replace("kN.mat", "")) in axial_forces,
        mat_files,
    )

    dataset = {}
    true_frs = {rpm: [] for rpm in [523, 937]}
    for f in mat_files:
        # print(f"Importing {f}...")
        data = loadmat(f)

        # Get specs from filename
        rpm, radial_load, axial_load = (
            f.stem.replace("rpm", "").replace("kN", "").split("_")
        )
        rpm = int(rpm)
        radial_load = float(radial_load)
        axial_load = float(axial_load)

        g_found = False
        rpm_found = False

        for i in range(5):
            try:
                signal = data[f"Signal_{i}"]
            except KeyError:
                continue

            unit = signal[0, 0]["y_values"]["quantity"][0, 0]["label"][0, 0][0]
            # print("Acceleration unit:", unit)

            if unit == "g":
                # Get sampling frequency from the increment of x_values, which is in seconds
                increment = signal[0, 0]["x_values"]["increment"][0, 0][0, 0]
                fs = 1 / increment

                # Signal not actually in g, but m/s^2, but since we are fine with m/s^2, wecan ignore the scaling factor
                acc_signal = signal[0, 0]["y_values"]["values"][0, 0][
                    :, bearing_index
                ]  # Last bearing (4th) is the one with the fault

                # Truncate signal if signal_max_len is specified (in seconds)
                if measurement_max_len is not None:
                    acc_signal = acc_signal[: int(measurement_max_len * fs)]

                g_found = True

            elif unit == "rpm":
                # Get true rotating frequency
                rpm_true = np.mean(signal[0, 0]["y_values"]["values"][0, 0])
                factor = signal[0, 0]["y_values"]["quantity"][0, 0][
                    "unit_transformation"
                ][0, 0]["factor"][0, 0][0, 0]
                fr = rpm_true * factor / 60
                true_frs[rpm].append(fr)

                rpm_found = True

            if g_found and rpm_found:
                break

        class_label = class_map_inverse[f.parent.name]
        dataset[(class_label, rpm, radial_load, axial_load)] = acc_signal

    for k, v in true_frs.items():
        true_frs[k] = np.mean(v)

    if verbose:
        print(f"Dataset size: {len(dataset)} measurements")
        print(f"Measurement duration: {len(acc_signal) / fs:.1f} s")
        print(f"Sampling frequency (fs): {fs / 1000} kHz")
        print(f"Rotating frequencies (frs):")
        print(
            "\n".join(
                [
                    f"  Nominal {rpm} rpm, true {fr_true * 60:.2f} rpm = {fr_true:.2f} Hz"
                    for rpm, fr_true in true_frs.items()
                ]
            ),
        )

    return dataset, int(fs), true_frs


def signal_windowing(signal, window_size, overlap):
    """
    Split a signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to be windowed.
    window_size : int
        The size of each window (number of samples).
    overlap : int
        The number of samples that overlap between consecutive windows.

    Returns
    -------
    np.ndarray
        An array of shape (num_windows, window_size) containing the windowed segments of the input signal.
    """

    step = max(1, int(window_size * (1 - overlap)))
    num_windows = 1 + (len(signal) - window_size) // step
    windows = np.array(
        [signal[i * step : i * step + window_size] for i in range(num_windows)]
    )

    return windows


def polito_to_sklearn_format(
    polito_dict,
    rpms=[],
    radial_forces=[],
    axial_forces=[],
    window_size=20480,
    # window_size=20480 // 4,
    overlap=0.9,
    verbose=True,
):
    """
    Convert the Polito dataset dictionary into NumPy arrays suitable for scikit-learn.

    Samples are added to the training and/or test sets based on the provided
    filter lists. Any filter list left empty means "accept all" for that field.
    A sample can appear in both sets if train and test filters overlap.

    Class labels are mapped as follows:
    - "H" -> 0
    - "IR" -> 1
    - "OR" -> 2

    Parameters
    ----------
    polito_dict : dict
        Dataset dictionary produced by `import_polito`.
    rpms : list, optional
        RPM values for all samples.
    radial_forces : list, optional
        Radial loads for all samples.
    axial_forces : list, optional
        Axial loads for all samples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(X, y)` where:
        - `X` contains windowed vibration samples
        - `y` contains class labels
    """

    X = []
    y = []
    X_rpm = []

    class_label_map = {"H": 0, "IR": 1, "OR": 2}

    for (class_label, rpm, radial_load, axial_load), signal in polito_dict.items():
        # Check which filters are specified (non-empty)
        rpm_check = not rpms or rpm in rpms
        radial_check = not radial_forces or radial_load in radial_forces
        axial_check = not axial_forces or axial_load in axial_forces

        windowed_signal = None
        if rpm_check and radial_check and axial_check:
            windowed_signal = signal_windowing(
                signal, window_size=window_size, overlap=overlap
            )

            X.append(windowed_signal)
            y.extend(
                [class_label_map[class_label]] * len(windowed_signal)
            )  # Extend with the same label for each window
            X_rpm.extend([rpm] * len(windowed_signal))

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    X_rpm = np.array(X_rpm)

    if verbose:
        print(
            f"Num samples: {len(X)}, sample size: {X.shape[1]}, class distribution: {np.bincount(y)}"
        )

    return X, y, X_rpm


def squared_envelope(x):
    """Compute the squared envelope of a signal using the Hilbert transform.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    squared_envelope : np.ndarray
        Squared envelope of the input signal.
    """

    analytic_signal = hilbert(x)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope**2

    return squared_envelope


def plot_confusion_matrix(clf, X_test, y_test):
    y_hat = clf.predict(X_test)

    acc = np.mean(y_hat == y_test)
    # print(f"Test accuracy: {acc*100:.2f}%")

    classes = np.unique(np.concatenate([y_test, y_hat]))
    cm = confusion_matrix(y_test, y_hat, labels=classes)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Healthy", "IR", "OR"]
    ).plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
