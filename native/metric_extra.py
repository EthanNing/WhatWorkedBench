import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def independent_metrics(family, data, prediction):
    if family == "graph_link_prediction":
        positive = prediction[data["test_labels"] == 1]
        negative = prediction[data["test_labels"] == 0]
        differences = positive[:, None] - negative[None, :]
        return {"native_value": float(np.mean((differences > 0) + .5 * (differences == 0)))}
    fs = int(data["sample_rate"])
    reference = data["true_events"]
    reference = reference[(reference >= fs) & (reference < len(data["signal"]) - fs)]
    differences = np.abs(prediction[:, None] - reference[None, :])
    metrics = {}
    for ms in (50, 100, 150):
        edges = csr_matrix((differences <= round(ms * fs / 1000)).astype(np.int8))
        matching = maximum_bipartite_matching(edges, perm_type="column")
        correct = int(sum(matching >= 0))
        denominator = len(prediction) + len(reference)
        metrics[str(ms)] = {"true_positive": correct, "f1": 2 * correct / denominator if denominator else 1.}
    return {"native_value": metrics["100"]["f1"], "tolerances": metrics}
