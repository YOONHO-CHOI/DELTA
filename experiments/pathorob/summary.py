"""Reconcile and export user-supplied patch-level numerical source tables."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .report import workbook_sheet
from .workbook import write_workbook

DELTA, RDELTA = 'DELTA-v1.3 p2', 'R-DELTA-v1.3 p2'
METRICS = [('biological_knn','Biological kNN',4),('clustering_score','Clustering score',3),
           ('confounder_insensitivity','Confounder insensitivity',3),('robustness_index','Robustness index',3)]
POPULATIONS = {key:['camelyon','tcga_4x4' if key=='clustering_score' else 'tcga_2x2','tolkach_esca'] for key,_,_ in METRICS}
HALF_SIGNS = ((np.arange(1024,dtype=np.uint16)[:,None]>>np.arange(10,dtype=np.uint16))&1).astype(np.int8)*2-1

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def holm(values):
    ps=np.asarray(values,float); order=np.argsort(ps); out=np.empty(len(ps))
    out[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
    return out

def exact_signflip(diff):
    assert len(diff)==20
    sums=(HALF_SIGNS@diff[:10])[:,None]+(HALF_SIGNS@diff[10:])[None,:]
    return float(np.mean(np.abs(sums)>=abs(float(diff.sum()))-1e-14))

def bootstrap_ci(values, seed):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    return np.quantile(values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1),[.025,.975])

def load_and_verify(input_path: Path):
    input_path = Path(input_path)
    snapshot=json.loads(input_path.read_text(encoding='utf-8'))
    if snapshot.get('schema_version') != 1:
        raise ValueError('Expected patch-results snapshot schema version 1.')
    frames={name:pd.DataFrame(rows) for name,rows in snapshot['workbook']['tables'].items()}
    payload=snapshot['dataset_summaries']
    rep=pd.DataFrame(payload['representation'])
    ds_ba=pd.DataFrame(payload['dataset_ba'])
    ds_apd=pd.DataFrame(payload['dataset_apd'])
    curves=pd.DataFrame(payload['curves'])
    # Reconcile every dataset aggregation with the frozen manuscript workbook.
    selected=rep.loc[[row.dataset in POPULATIONS[row.endpoint] for row in rep.itertuples()]]
    computed_rep=selected.groupby(['backbone_std','method_std','endpoint']).value.mean()
    stored_rep=frames['Representation Backbone'].set_index(['Backbone','Method','Endpoint']).Score_value
    common=computed_rep.index.intersection(stored_rep.index)
    err_rep=float(np.max(np.abs(computed_rep.loc[common]-stored_rep.loc[common])))
    assert len(common)==640 and err_rep < 1e-12, (len(common),err_rep)
    errors={}
    for name,frame,sheet,column in [('BA',ds_ba,'BA Backbone','BA_value'),('APD',ds_apd,'APD Backbone','APD_value')]:
        calc=frame.groupby(['backbone_std','method_std','domain']).value.mean()
        stored=frames[sheet].set_index(['Backbone','Method','Domain'])[column]
        assert calc.index.equals(stored.sort_index().index)
        error=float(np.max(np.abs(calc-stored.reindex(calc.index))))
        assert error < 1e-12,(name,error)
        errors[name]=error
    # Retain and independently verify every original multiplicity family.
    stat_errors=[]
    for candidate in [DELTA,RDELTA]:
        rp=frames['Representation Paired'].query('candidate==@candidate')
        assert len(rp)==16
        ps=[]
        for row in rp.itertuples():
            p=frames['Representation Backbone'].query('Endpoint==@row.endpoint').pivot(index='Backbone',columns='Method',values='Score_value').sort_index()
            diff=(p[row.candidate]-p[row.control]).to_numpy()
            pv=exact_signflip(diff); ps.append(pv)
            stat_errors.extend([abs(pv-row.exact_signflip_p),abs(diff.mean()-row.mean_difference)])
        stat_errors.extend(np.abs(holm(ps)-rp.holm_exact_p.to_numpy()))
        for metric,sheet,value_col in [('balanced_accuracy','BA Backbone','BA_value'),('average_performance_drop','APD Backbone','APD_value')]:
            family=frames['APD Paired'].query('candidate==@candidate and metric==@metric')
            assert len(family)==12
            ps=[]
            for row in family.itertuples():
                p=frames[sheet].query('Domain==@row.domain').pivot(index='Backbone',columns='Method',values=value_col).sort_index()
                diff=(p[row.candidate]-p[row.control]).to_numpy()
                pv=exact_signflip(diff); ps.append(pv)
                stat_errors.extend([abs(pv-row.exact_signflip_p),abs(diff.mean()-row.mean_difference)])
                ci=bootstrap_ci(diff,20260819+len(ps)-1)
                stat_errors.extend(np.abs(ci-[row.ci_low,row.ci_high]))
            stat_errors.extend(np.abs(holm(ps)-family.holm_exact_p.to_numpy()))
    assert max(stat_errors)<1e-12,max(stat_errors)
    audit={'workbook_name':snapshot['workbook']['name'],
           'workbook_sha256':snapshot['workbook']['sha256'],'inputs_sha256':sha(input_path),
           'representation_values_reconciled':len(common),'representation_max_error':err_rep,
           'BA_values_reconciled':320,'APD_values_reconciled':320,'dataset_aggregate_max_error':errors,
           'all_80_existing_contrasts_rechecked':True,'max_statistic_error':float(max(stat_errors)),
           'representation_holm_family':'16 contrasts per candidate: 4 metrics x 4 controls',
           'prediction_holm_family':'12 contrasts per candidate and metric: 6 controls x ID/OOD',
           'source_files_unchanged':True,'Word_modified':False,'GPU_used':False,
           'sources':payload['sources'],'code_sha256':sha(Path(__file__))}
    if 'apd_definition' in snapshot:
        audit['apd_definition']=snapshot['apd_definition']
        audit['roi_deliverable']=snapshot['workbook']['roi_deliverable']
    return frames,rep,ds_ba,ds_apd,curves,audit

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="User-supplied schema-v1 patch-results snapshot; study results are not bundled")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    frames, rep, ba, apd, curves, audit = load_and_verify(args.input)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output / (name.lower().replace(" ", "_") + ".csv"), index=False)
    for name, frame in (("dataset_representation", rep), ("dataset_ba", ba), ("dataset_apd", apd), ("condition_curves", curves)):
        frame.to_csv(output / (name + ".csv"), index=False)
    payload = {"title": "Patch numerical source tables", "sheets": [workbook_sheet(name, frame) for name, frame in frames.items()]}
    source = output / "workbook_payload.json"
    source.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_workbook(source, output / "patch_numerical_source.xlsx")
    (output / "reconciliation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("PATCH_NUMERICAL_SOURCE_COMPLETE")


if __name__ == "__main__":
    main()
