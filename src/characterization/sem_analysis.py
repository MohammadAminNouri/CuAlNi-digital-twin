"""SEM preprocessing, unsupervised regions, and provenance-gated phase inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

ALLOWED_PHASE_CLASSES = ("martensite", "beta", "gamma2")


class SEMAnalysisError(ValueError):
    pass


@dataclass
class SEMAnalysisResult:
    processed_image: np.ndarray
    label_map: np.ndarray
    area_fractions_2d: dict[str, float]
    segmentation_mode: str
    classification: str
    confidence: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "Pixel/area fractions from one 2D field are not thermodynamic phase fractions.",
        "SEM intensity depends on detector, topography, orientation, charging, and acquisition settings.",
        "Unsupervised intensity/texture regions must not be interpreted as phases without labelled validation.",
    )

    def to_dict(self) -> dict[str, Any]:
        # Report/UI previews are deliberately bounded.  Full-resolution arrays
        # remain available on the Python result object for quantitative follow-up,
        # while JSON exports avoid ballooning with multi-megapixel micrographs.
        maximum_preview_dimension = 320
        stride = max(
            1,
            int(np.ceil(max(self.processed_image.shape) / maximum_preview_dimension)),
        )
        processed_preview = self.processed_image[::stride, ::stride]
        label_preview = self.label_map[::stride, ::stride]
        return {
            "classification": self.classification,
            "segmentation_mode": self.segmentation_mode,
            "area_fractions_2d": self.area_fractions_2d,
            "confidence": self.confidence,
            "limitations": list(self.limitations),
            "image_shape": list(self.processed_image.shape),
            "preview": {
                "processed_image": np.round(processed_preview, decimals=5).tolist(),
                "label_map": label_preview.astype(int).tolist(),
                "label_names": list(self.area_fractions_2d),
                "downsample_stride": stride,
                "preview_shape": list(processed_preview.shape),
                "purpose": "visual quality-control preview; full-resolution arrays remain on the Python result",
            },
        }


def load_sem_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
    else:
        array = np.asarray(Image.open(image))
    if array.ndim == 3:
        rgb = array[..., :3].astype(float)
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if array.ndim != 2:
        raise SEMAnalysisError("SEM image must be a 2D grayscale or RGB array")
    array = array.astype(float)
    finite = np.isfinite(array)
    if not finite.any():
        raise SEMAnalysisError("SEM image contains no finite pixels")
    fill = float(np.median(array[finite]))
    array[~finite] = fill
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        raise SEMAnalysisError("SEM image has insufficient intensity contrast")
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def preprocess_sem(image: str | Path | np.ndarray, *, denoise_sigma_px: float = 1.0) -> np.ndarray:
    normalized = load_sem_image(image)
    if denoise_sigma_px < 0:
        raise SEMAnalysisError("denoise_sigma_px cannot be negative")
    return ndimage.gaussian_filter(normalized, sigma=denoise_sigma_px)


def pixel_features(processed: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    gradient_x = ndimage.sobel(processed, axis=1)
    gradient_y = ndimage.sobel(processed, axis=0)
    gradient = np.hypot(gradient_x, gradient_y)
    local_mean = ndimage.uniform_filter(processed, size=9)
    local_variance = np.maximum(
        ndimage.uniform_filter(processed**2, size=9) - local_mean**2, 0.0
    )
    features = np.column_stack(
        [processed.ravel(), gradient.ravel(), local_mean.ravel(), np.sqrt(local_variance).ravel()]
    )
    return features, processed.shape


def _validate_classifier_metadata(metadata: Mapping[str, Any] | None) -> tuple[str, ...]:
    if metadata is None:
        raise SEMAnalysisError("Classifier metadata with provenance and validation is required")
    required = ("source", "training_dataset_sha256", "class_labels", "validation_metrics")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise SEMAnalysisError("Classifier metadata is missing: " + ", ".join(missing))
    labels = tuple(str(label) for label in metadata["class_labels"])
    if not labels or not set(labels).issubset(ALLOWED_PHASE_CLASSES):
        raise SEMAnalysisError("Classifier labels must be a subset of martensite, beta, gamma2")
    return labels


def analyze_sem(
    image: str | Path | np.ndarray,
    *,
    classifier: Any | None = None,
    classifier_metadata: Mapping[str, Any] | None = None,
    n_regions: int = 3,
    denoise_sigma_px: float = 1.0,
    random_state: int = 42,
) -> SEMAnalysisResult:
    processed = preprocess_sem(image, denoise_sigma_px=denoise_sigma_px)
    features, shape = pixel_features(processed)
    if classifier is None:
        if not 2 <= n_regions <= 8:
            raise SEMAnalysisError("n_regions must lie between 2 and 8")
        model = MiniBatchKMeans(
            n_clusters=n_regions,
            batch_size=min(4096, len(features)),
            n_init=10,
            random_state=random_state,
        )
        raw_labels = model.fit_predict(features)
        intensity_centers = model.cluster_centers_[:, 0]
        order = np.argsort(intensity_centers)
        remap = np.empty_like(order)
        remap[order] = np.arange(n_regions)
        labels = remap[raw_labels].reshape(shape)
        names = [f"region_{index + 1}" for index in range(n_regions)]
        mode = "unsupervised_intensity_texture_regions"
        classification = "experimental_data_analysis"
        confidence = {
            "phase_classification_available": False,
            "reason": "No validated, provenance-documented SEM phase classifier supplied",
            "inertia": float(model.inertia_),
        }
    else:
        names = list(_validate_classifier_metadata(classifier_metadata))
        raw_labels = np.asarray(classifier.predict(features))
        if raw_labels.shape != (len(features),):
            raise SEMAnalysisError("Classifier must return one label per pixel")
        if np.issubdtype(raw_labels.dtype, np.number):
            if raw_labels.min() < 0 or raw_labels.max() >= len(names):
                raise SEMAnalysisError("Numeric classifier labels are outside metadata class_labels")
            label_names = np.array(names, dtype=object)[raw_labels.astype(int)]
        else:
            label_names = raw_labels.astype(str)
            if not set(np.unique(label_names)).issubset(names):
                raise SEMAnalysisError("Classifier returned labels absent from metadata")
        name_to_index = {name: index for index, name in enumerate(names)}
        labels = np.vectorize(name_to_index.__getitem__)(label_names).reshape(shape)
        mode = "supervised_phase_classifier"
        classification = "machine_learning_prediction"
        confidence = {
            "phase_classification_available": True,
            "classifier_provenance": dict(classifier_metadata or {}),
            "per_pixel_probability_available": hasattr(classifier, "predict_proba"),
        }
    fractions = {
        name: float(np.mean(labels == index)) for index, name in enumerate(names)
    }
    return SEMAnalysisResult(processed, labels, fractions, mode, classification, confidence)
