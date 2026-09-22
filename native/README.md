# Native CPU replay

This standalone pack executes all 36 task conditions from prepared inputs and checks 1,248 configuration records against saved predictions and independently computed metrics.

```bash
python -m pip install -r native/requirements.txt
python native/replay.py --output output/native-audit.json
```

Create `output/` first. To replay one task, pass its name, for example `python native/replay.py scifact --output output/scifact-audit.json`.

The runner uses pinned numerical packages, one computation thread, and no model API calls or source downloads. Prepared inputs, reference labels, and full outcomes belong to the evaluator. The code checks original file identities, utility values, native metrics, prediction hashes, and numerical warnings. Native inputs preserve the recorded preparation; source acquisition and preparation details are retained in `../benchmark/sources/`.

Prepared data retain upstream terms. SciFact claims and annotations use CC BY 4.0 and its corpus uses ODC-By 1.0. MIT-BIH uses ODC-By 1.0. Newman network sources state free scientific use; retain their citations and consult the source terms for other uses. Full per-source notices are in [data sources](../guides/data-sources.md). Project code uses Apache 2.0.

The six paired variants contribute 96 repeated original-subcube outcomes. These records retain their paired identity. Reproduction across operating systems and BLAS implementations can change exact numerical hashes; use the pinned environment for the strict replay.
