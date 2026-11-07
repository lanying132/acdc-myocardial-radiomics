"""Reproduce the ACDC study from a local, unmodified database directory."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ["NOR", "DCM", "HCM", "MINF", "RV"]
FEATURES = ["mean", "sd", "p10", "p90", "contrast", "homogeneity", "entropy"]
BASELINE = ["LVEDVi", "LVESVi", "LVEF", "RVEDVi", "RVESVi", "RVEF",
            "LVMi", "ED_thickness_p90", "ED_thickness_max", "thickening_p10",
            "thickening_sd"]


def load_phase(folder, frame):
    prefix = folder / f"{folder.name}_frame{int(frame):02}"
    image = nib.load(str(prefix) + ".nii.gz")
    mask = nib.load(str(prefix) + "_gt.nii.gz")
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-4):
        raise ValueError(f"Image and mask grids differ: {prefix}")
    pixels = image.get_fdata(dtype=np.float32)
    labels = np.asarray(mask.dataobj)
    if pixels.ndim != 3 or not np.isfinite(pixels).all() or set(np.unique(labels)) != {0, 1, 2, 3}:
        raise ValueError(f"Invalid cine image or reference mask: {prefix}")
    spacing = nib.affines.voxel_sizes(image.affine)
    return pixels, labels, spacing


def thickness_profile(labels, spacing):
    """Approximate sector thickness with rays in the central LV slices."""
    cavity = labels == 3
    slices = np.flatnonzero(cavity.sum(axis=(0, 1)) >= 40)
    if len(slices) < 2:
        raise ValueError("Insufficient LV coverage")
    slices = slices[1:-1] if len(slices) >= 4 else slices
    angles = np.arange(72) * (2 * np.pi / 72)
    radii = np.arange(0, 65, 0.5)
    offsets = np.stack([np.cos(angles) / spacing[0], np.sin(angles) / spacing[1]])
    profiles = []
    for z in slices:
        center = np.array(ndi.center_of_mass(cavity[:, :, z]))
        points = center[:, None, None] + offsets[:, :, None] * radii
        rays = ndi.map_coordinates(labels[:, :, z], points, order=0)
        thickness = []
        for ray in rays:
            exits = np.flatnonzero(ray != 3)
            if not exits.size or ray[exits[0]] != 2:
                thickness.append(np.nan)
                continue
            end = np.flatnonzero(ray[exits[0]:] != 2)
            thickness.append(0.5 * end[0] if end.size else np.nan)
        profiles.append(thickness)
    # Per-angle summaries limit the influence of individual basal/apical cuts.
    profile = np.full(72, np.nan)
    for angle, values in enumerate(np.asarray(profiles).T):
        valid = values[np.isfinite(values)]
        if valid.size:
            profile[angle] = np.median(valid)
    return profile


def conventional(ed, es, height, weight):
    bsa = np.sqrt(height * weight / 3600)
    output = []
    for label in (3, 1):
        volumes = [np.count_nonzero(phase[1] == label) * np.prod(phase[2]) / 1000
                   for phase in (ed, es)]
        start, end = volumes
        output.extend([start / bsa, end / bsa, (start - end) / start * 100])
    mass = np.count_nonzero(ed[1] == 2) * np.prod(ed[2]) / 1000 * 1.05 / bsa
    ed_profile, es_profile = (thickness_profile(phase[1], phase[2]) for phase in (ed, es))
    thickening = es_profile - ed_profile
    return output + [mass, np.nanpercentile(ed_profile, 90), np.nanmax(ed_profile),
                     np.nanpercentile(thickening, 10), np.nanstd(thickening)]


def texture(pixels, roi):
    levels = 16
    gray = np.clip(np.floor((pixels + 3) * levels / 6), 0, levels - 1).astype(int)
    pairs = np.zeros((levels, levels))
    # Build a pooled symmetric GLCM using only pairs wholly inside the myocardium.
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        left = (slice(max(0, -dx), pixels.shape[0] - max(0, dx)),
                slice(max(0, -dy), pixels.shape[1] - max(0, dy)))
        right = (slice(max(0, dx), pixels.shape[0] - max(0, -dx)),
                 slice(max(0, dy), pixels.shape[1] - max(0, -dy)))
        inside = roi[left] & roi[right]
        codes = gray[left][inside] * levels + gray[right][inside]
        pairs += np.bincount(codes, minlength=levels**2).reshape(levels, levels)
    if not pairs.sum():
        return None
    prob = (pairs + pairs.T) / (2 * pairs.sum())
    distance = np.subtract.outer(np.arange(levels), np.arange(levels)) ** 2
    positive = prob[prob > 0]
    values = pixels[roi]
    return [values.mean(), values.std(), *np.percentile(values, [10, 90]),
            (prob * distance).sum(), (prob / (1 + distance)).sum(),
            -(positive * np.log2(positive)).sum()]


def radiomics(phase):
    image, labels, spacing = phase
    factors = spacing[:2] / 1.5
    structure = ndi.generate_binary_structure(2, 1)
    summaries = [[], [], []]
    for z in range(image.shape[2]):
        plane = ndi.zoom(image[:, :, z], factors, order=1, grid_mode=True, mode="nearest")
        roi = ndi.zoom(labels[:, :, z] == 2, factors, order=0, grid_mode=True, mode="nearest")
        if roi.sum() < 50:
            continue
        low, high = np.percentile(plane, [1, 99])
        normalized = np.clip(plane, low, high)
        scale = normalized.std()
        if scale == 0:
            continue
        normalized = (normalized - normalized.mean()) / scale
        for bucket, region in zip(summaries, [roi, ndi.binary_erosion(roi, structure),
                                             ndi.binary_dilation(roi, structure)]):
            if region.sum() >= 20:
                summary = texture(normalized, region)
                if summary is not None:
                    bucket.append(summary)
    if any(not bucket for bucket in summaries):
        raise ValueError("No eligible myocardium for one or more mask variants")
    return np.asarray([np.median(bucket, axis=0) for bucket in summaries])


def extract(database):
    rows = []
    for split in ("training", "testing"):
        for folder in sorted((database / split).glob("patient*")):
            metadata = dict(line.split(": ", 1) for line in
                            (folder / "Info.cfg").read_text().splitlines() if line.strip())
            ed, es = (load_phase(folder, metadata[key]) for key in ("ED", "ES"))
            height, weight = (float(metadata[key]) for key in ("Height", "Weight"))
            base = conventional(ed, es, height, weight)
            if not np.isfinite(base).all():
                raise ValueError(f"Nonfinite conventional measurements: {folder}")
            rows.append((split, folder.name, metadata["Group"], base, radiomics(ed), radiomics(es)))
    return rows


def score(true, predicted):
    matrix = confusion_matrix(true, predicted, labels=CLASSES)
    correct = matrix.diagonal()
    recall = correct / matrix.sum(axis=1)
    f1 = 2 * correct / (matrix.sum(axis=1) + matrix.sum(axis=0))
    return {"macro_f1": float(f1.mean()), "accuracy": float(correct.sum() / matrix.sum()),
            "recall": recall.tolist(), "f1": f1.tolist(), "matrix": matrix.tolist()}


def paired_interval(true, first, second):
    rng = np.random.default_rng(8122)
    groups = [np.flatnonzero(true == name) for name in CLASSES]
    differences = []
    for _ in range(5000):
        indexes = np.concatenate([rng.choice(group, group.size, replace=True) for group in groups])
        differences.append(score(true[indexes], second[indexes])["macro_f1"]
                           - score(true[indexes], first[indexes])["macro_f1"])
    return np.percentile(differences, [2.5, 97.5]).tolist()


def fit(train, test, train_labels, test_labels, alternatives):
    pipeline = make_pipeline(VarianceThreshold(), StandardScaler(),
                             LogisticRegression(max_iter=2000))
    grid = {"logisticregression__C": [0.01, 0.1, 1, 10]}
    inner = StratifiedKFold(4, shuffle=True, random_state=8122)
    search = GridSearchCV(pipeline, grid, cv=inner, scoring="f1_macro", error_score="raise")
    outer = StratifiedKFold(5, shuffle=True, random_state=2026)
    validation = []
    for build, evaluate in outer.split(train, train_labels):
        model = clone(search).fit(train[build], train_labels[build])
        validation.append(score(train_labels[evaluate],
                                model.predict(train[evaluate]))["macro_f1"])
    search.fit(train, train_labels)
    predicted = search.predict(test)
    result = score(test_labels, predicted)
    result["nested_cv_f1"] = validation
    result["selected_c"] = search.best_params_["logisticregression__C"]
    result["perturbations"] = {}
    for name, alternative in alternatives.items():
        changed = search.predict(alternative)
        result["perturbations"][name] = {"macro_f1": score(test_labels, changed)["macro_f1"],
                                         "changed": int(np.count_nonzero(changed != predicted))}
    if alternatives:
        estimator = search.best_estimator_
        result["coefficients"] = estimator[-1].coef_.tolist()
        result["included"] = estimator[0].get_support().tolist()
        result["coefficient_classes"] = estimator[-1].classes_.tolist()
    return result, predicted


def analyze(rows):
    training = np.array([row[0] == "training" for row in rows])
    labels = np.array([row[2] for row in rows])
    if len({row[1] for row in rows}) != len(rows):
        raise ValueError("Duplicate patient identifiers")
    for subset, expected in [(training, 20), (~training, 10)]:
        if any(np.count_nonzero(labels[subset] == label) != expected for label in CLASSES):
            raise ValueError("Unexpected ACDC cohort counts")
    baseline = np.array([row[3] for row in rows])
    ed, es = (np.array([row[index] for row in rows]) for index in (4, 5))
    variants = {
        "A": [baseline] * 3,
        "B": [ed[:, index] for index in range(3)],
        "C": [np.column_stack((baseline, ed[:, index])) for index in range(3)],
        "C_ED_ES": [np.column_stack((baseline, ed[:, index], es[:, index]))
                    for index in range(3)],
    }
    models, predictions = {}, {}
    for name, (reference, eroded, dilated) in variants.items():
        alternatives = {} if name == "A" else {"erosion": eroded[~training],
                                                "dilation": dilated[~training]}
        models[name], predictions[name] = fit(reference[training], reference[~training],
                                               labels[training], labels[~training], alternatives)
    truth = labels[~training]
    a, c = predictions["A"], predictions["C"]
    return {
        "versions": {name: version(name) for name in ("numpy", "scipy", "nibabel", "scikit-learn")},
        "classes": CLASSES, "features": FEATURES, "baseline": BASELINE, "models": models,
        "difference": models["C"]["macro_f1"] - models["A"]["macro_f1"],
        "confidence_interval": paired_interval(truth, a, c),
        "error_changes": {label: {
            "corrected": int(((truth == label) & (a != truth) & (c == truth)).sum()),
            "new": int(((truth == label) & (a == truth) & (c != truth)).sum()),
        } for label in CLASSES},
        "cohort": {split: {label: np.percentile(baseline[(training if split == "training"
                            else ~training) & (labels == label)], [25, 50, 75], axis=0).tolist()
                           for label in CLASSES} for split in ("training", "testing")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = extract(args.database)
    print(f"Extracted {len(rows)} patient records.", flush=True)
    result = analyze(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
