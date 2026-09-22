"""Standalone native replay: no project imports, API, or source downloads."""
import argparse
import gzip
import hashlib
import json
import math
import time
import warnings
from importlib.metadata import version
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from native_workflows import evaluate as evaluate_four
from native_six import evaluate as evaluate_six
from metric_reference import independent_metric
from native_extra import predict as predict_extra, evaluate as evaluate_extra
from metric_extra import independent_metrics
EXTRA_FAMILIES = ('beat_detection', 'graph_link_prediction')

def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def run(names, output):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text())
    for filename, expected in manifest['files_sha256'].items():
        if digest((root/filename).read_bytes()) != expected:
            raise ValueError('Pack file changed: '+filename)
    for package, expected in manifest['replay_versions'].items():
        if version(package) != expected:
            raise ValueError('Use the declared replay package version: '+package)
    if names and set(names)-set(manifest['instances']):
        raise ValueError('Unknown instance')
    rows = []; started = time.monotonic()
    for name in names or manifest['instances']:
        with gzip.open(root/'inputs'/manifest['conditions'][name]['source_instance']/'metadata.json.gz','rt') as stream:
            data = json.load(stream)
        with np.load(root/'inputs'/manifest['conditions'][name]['source_instance']/'arrays.npz', allow_pickle=False) as arrays:
            data.update({key:arrays[key] for key in arrays})
        table = json.loads((root/'references'/(name+'.json')).read_text())
        family = table['manifest']['source']['family']
        for mask, saved in table['outcomes'].items():
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                if family in EXTRA_FAMILIES:
                    prediction = predict_extra(family, data, mask)
                    result = evaluate_extra(family, data, prediction)
                else:
                    prediction, result = (evaluate_four if len(mask) == 4 else evaluate_six)(family, data, mask)
            pred_hash = digest(prediction.tolist()) if family in EXTRA_FAMILIES else digest(prediction if isinstance(prediction,dict) else prediction.tobytes())
            if family in EXTRA_FAMILIES:
                checked = independent_metrics(family, data, prediction)
                native_reference = checked['native_value']
                for ms, values in checked.get('tolerances', {}).items():
                    original = saved['matching_tolerance_sensitivity'][ms]
                    assert values['true_positive'] == original['true_positive'] and values['f1'] == original['f1']
            else:
                native_reference = independent_metric(family, data, prediction, saved)
            assert pred_hash == saved['prediction_sha256'], (name,mask,'prediction')
            assert result['utility'] == saved['utility'], (name,mask,'utility')
            assert math.isclose(native_reference,saved['native_value'],abs_tol=1e-10,rel_tol=1e-12), (name,mask,'metric')
            warnings_now = [{'category':w.category.__name__,'message':str(w.message)} for w in captured]
            assert warnings_now == saved['warnings'], (name,mask,'warnings')
            rows.append({'instance':name,'family':family,'mask':mask,'passed':True})
        print(json.dumps({'instance':name,'passed':len(table['outcomes'])}),flush=True)
    report = {'passed':len(rows),'task_conditions':len(names or manifest['instances']), 'distinct_source_instances':len({manifest['conditions'][name]['source_instance'] for name in names or manifest['instances']}),
              'wall_seconds':time.monotonic()-started,'rows':rows,
              'offline':True,'project_imports':False,'prepared_inputs_not_raw_source_acquisition':True}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instances',nargs='*')
    parser.add_argument('--output',default='native_replay_audit.json')
    args=parser.parse_args();run(args.instances,args.output)
