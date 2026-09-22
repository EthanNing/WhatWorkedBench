"""Native six-option signal and graph workflows with isolated evaluation labels."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def switches(mask):
    if not isinstance(mask, str) or len(mask) != 6 or set(mask) - {"0", "1"}:
        raise ValueError("Expected a six-bit configuration")
    return tuple(v == "1" for v in mask)


def detect_events(signal, sample_rate, mask):
    a, b, c, d, e, f = switches(mask)
    signal = np.asarray(signal, dtype=np.float64)
    work = signal - np.median(signal)
    if a:
        work = sosfiltfilt(butter(2, .5, btype="highpass", fs=sample_rate, output="sos"), work)
    if b:
        work = sosfiltfilt(butter(2, 35., btype="lowpass", fs=sample_rate, output="sos"), work)
    envelope = np.diff(work, prepend=work[0]) ** 2 if c else np.abs(work)
    if d:
        envelope = uniform_filter1d(envelope, size=max(1, round(.120 * sample_rate)), mode="nearest")
    if e:
        center = np.median(envelope)
        height = center + 3 * 1.4826 * np.median(np.abs(envelope - center))
    else:
        height = np.mean(envelope) + np.std(envelope)
    candidates, _ = find_peaks(envelope, height=height,
                               distance=max(1, round((.300 if f else .200) * sample_rate)))
    radius = round(.075 * sample_rate)
    aligned = []
    for candidate in candidates:
        low, high = max(0, candidate - radius), min(len(work), candidate + radius + 1)
        aligned.append(low + int(np.argmax(np.abs(work[low:high]))))
    # This offline workflow evaluates an interior interval to avoid filter edges.
    result = np.array(sorted({p for p in aligned if sample_rate <= p < len(signal) - sample_rate}), dtype=np.int64)
    if not np.isfinite(work).all():
        raise ValueError("Nonfinite filtered signal")
    return result


def predict_links(n_nodes, observed_edges, fit_pairs, fit_labels, test_pairs, mask):
    import networkx as nx
    enabled = switches(mask)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(n_nodes)))
    graph.add_edges_from((int(u), int(v)) for u, v in observed_edges)
    pairs = [(int(u), int(v)) for u, v in np.concatenate([fit_pairs, test_pairs])]
    columns = []
    if enabled[0]:
        columns.append(np.array([len(list(nx.common_neighbors(graph, u, v))) for u, v in pairs], dtype=float))
    for active, algorithm in zip(enabled[1:5], (nx.jaccard_coefficient, nx.adamic_adar_index,
                                              nx.resource_allocation_index, nx.preferential_attachment)):
        if active:
            columns.append(np.array([value for _, _, value in algorithm(graph, pairs)], dtype=float))
    features = np.column_stack(columns) if columns else np.zeros((len(pairs), 1))
    train, test = features[:len(fit_pairs)], features[len(fit_pairs):]
    if enabled[5]:
        scaler = StandardScaler().fit(train)
        train, test = scaler.transform(train), scaler.transform(test)
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=20260907)
    model.fit(train, fit_labels)
    result = model.predict_proba(test)[:, 1]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite link probabilities")
    return result


def predict(family, data, mask):
    # Only the following explicit input fields can enter native prediction.
    if family == "beat_detection":
        return detect_events(data["signal"], int(data["sample_rate"]), mask)
    if family == "graph_link_prediction":
        return predict_links(int(data["n_nodes"]), data["observed_edges"], data["fit_pairs"],
                             data["fit_labels"], data["test_pairs"], mask)
    raise ValueError("Unknown native family")


def event_metric(prediction, reference, tolerance):
    prediction, reference = sorted(prediction), sorted(reference)
    left = right = true_positive = 0
    while left < len(prediction) and right < len(reference):
        if prediction[left] < reference[right] - tolerance:
            left += 1
        elif prediction[left] > reference[right] + tolerance:
            right += 1
        else:
            true_positive += 1
            left += 1
            right += 1
    return {"true_positive": true_positive, "predicted_events": len(prediction), "reference_events": len(reference),
            "precision": true_positive / len(prediction) if prediction else 0.,
            "recall": true_positive / len(reference) if reference else 0.,
            "f1": 2 * true_positive / (len(prediction) + len(reference)) if prediction or reference else 1.}


def evaluate(family, data, prediction):
    if family == "beat_detection":
        sample_rate = int(data["sample_rate"])
        reference = data["true_events"]
        reference = reference[(reference >= sample_rate) & (reference < len(data["signal"]) - sample_rate)]
        details = {str(ms): event_metric(prediction.tolist(), reference.tolist(), round(ms * sample_rate / 1000))
                   for ms in (50, 100, 150)}
        value = details["100"]["f1"]
        return {"native_metric": "event_F1_at_100ms", "native_value": value, "utility": value,
                "matching_tolerance_sensitivity": details}
    if family == "graph_link_prediction":
        value = float(roc_auc_score(data["test_labels"], prediction))
        return {"native_metric": "ROC_AUC", "native_value": value, "utility": value}
    raise ValueError("Unknown native family")
