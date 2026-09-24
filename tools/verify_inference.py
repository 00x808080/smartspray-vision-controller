#!/usr/bin/env python3
"""Freeze samples/gates, then measure layered real-model Python/C++ parity."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import subprocess
import numpy as np
import cv2
try:
    from . import onnx_deploy as ref
except ImportError:
    import onnx_deploy as ref


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def freeze(args):
    if args.output.exists():
        raise ValueError("manifest output must be new; never reselect after parity")
    derived = args.derived.resolve()
    images = {r["image_id"]: r for r in read_jsonl(derived/"images.jsonl") if r["split"] == "val"}
    objects = [r for r in read_jsonl(derived/"provenance.jsonl") if r["split"] == "val"]
    selection = json.loads((derived/"selection.json").read_text())
    fixed = json.loads(args.baseline_examples.read_text())["images"]
    reasons = defaultdict(list)
    def add(i, reason):
        if i not in images:
            raise ValueError("unknown frozen image")
        reasons[i].append(reason)
    for i in selection["ordinary_val_ids"]:
        add(i, "M2.2 ordinary fixed example")
    for row in fixed:
        add(row["image_id"], "M2.2 inspected example")
    for cls in (1,2):
        group = [r for r in objects if r["class_id"] == cls]
        for pred, name in [(lambda r: True,"smallest box"), (lambda r: r["partial_semantic"],"smallest partial"),
                           (lambda r: r["touches_image_edge"],"smallest edge")]:
            row = min((r for r in group if pred(r)), key=lambda r: ((r["bbox"][2]-r["bbox"][0])*(r["bbox"][3]-r["bbox"][1]), r["image_id"], r["raw_instance_id"]))
            add(row["image_id"], f"{ref.NAMES[cls-1]} {name}")
    ids = sorted(images)
    for j in np.linspace(0,len(ids)-1,32,dtype=int):
        if len(reasons) >= 32:
            break
        add(ids[j], "evenly spaced sorted validation IDs")
    for i in ids:
        if len(reasons) >= 32:
            break
        if i not in reasons:
            add(i, "sorted fill")
    real=[]
    selected_objects=[o for o in objects if o["image_id"] in reasons]
    for i in sorted(reasons):
        path=derived/"images/val"/images[i]["filename"]
        digest=ref.sha256(path)
        if digest!=images[i]["rgb_sha256"]:
            raise ValueError("M2.2 source image hash mismatch")
        real.append({"id":i,"path":str(path),"sha256":digest,"kind":"real","reasons":reasons[i]})
    fixtures=args.output.parent/"fixtures"; fixtures.mkdir(parents=True,exist_ok=True)
    synthetic=[]
    for name,w,h in [("no_resize_color",1024,1024),("odd_padding",1024,701),
                      ("resized_odd",1001,667),("round_tie_even",2048,1333),
                      ("round_tie_odd",2048,1335),("small_nonsquare",33,17)]:
        yy,xx=np.indices((h,w))
        bgr=np.stack([(xx*7+yy*11)%256,(xx*19+yy*3)%256,(xx*5+yy*29)%256],axis=2).astype(np.uint8)
        path=fixtures/(name+".png")
        if not cv2.imwrite(str(path),bgr): raise RuntimeError("fixture write failed")
        synthetic.append({"id":name,"path":str(path.resolve()),"sha256":ref.sha256(path),
                          "kind":"synthetic","width":w,"height":h,"reason":"numerical preprocessing fixture; no detector quality claim"})
    manifest={"schema_version":1,"selection_frozen_before_comparisons":True,"real_count":len(real),
              "samples":real+synthetic,"gates":ref.GATES,"deployment":ref.DEPLOYMENT,
              "runtime":ref.RUNTIME,"controller_geometry":ref.GEOMETRY,
              "coverage":{"classes":dict(Counter(o["class_name"] for o in selected_objects)),
                          "partial":sum(o["partial_semantic"] for o in selected_objects),
                          "edge":sum(o["touches_image_edge"] for o in selected_objects),
                          "tiny_area_le_16":sum((o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1])<=16 for o in selected_objects)},
              "matching":"maximum-cardinality one-to-one same-class IoU>=0.5; edges ordered by decreasing IoU, then candidate index; cross-class unmatched IoU>=0.5 recorded separately"}
    ref.write_json(args.output,manifest)
    print(f"Frozen {len(real)} real + {len(synthetic)} synthetic samples: {ref.sha256(args.output)}")


def error_stats(a,b,atol,rtol=0):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("shape/finiteness mismatch")
    delta=np.abs(a-b)
    return {"max_abs":float(delta.max(initial=0)),"mean_abs":float(delta.mean()) if delta.size else 0.,
            "elements":int(delta.size),"changed":int(np.count_nonzero(delta)),
            "over_budget":int(np.count_nonzero(delta>atol+rtol*np.abs(a))),
            "pass":bool(np.all(delta<=atol+rtol*np.abs(a)))}


def match(a,b):
    # Augmenting paths avoid greedy array-position matches and maximize cardinality.
    edges={}
    for i,x in enumerate(a):
        edges[i]=sorted([j for j,y in enumerate(b) if x["class_id"]==y["class_id"] and ref.iou(x["xyxy"],y["xyxy"])>=.5],
                        key=lambda j:(-ref.iou(x["xyxy"],b[j]["xyxy"]),b[j].get("candidate_index",j),j))
    owner={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or visit(owner[j],seen):
                owner[j]=i
                return True
        return False
    for i in sorted(edges,key=lambda i:(a[i].get("candidate_index",i),i)):
        visit(i,set())
    pairs=sorted((i,j) for j,i in owner.items())
    ua=sorted(set(range(len(a)))-{i for i,j in pairs})
    ub=sorted(set(range(len(b)))-{j for i,j in pairs})
    return pairs,ua,ub


def compare_detections(a,b):
    a,b=a["detections"],b["detections"]
    pairs,ua,ub=match(a,b)
    box=error_stats([a[i]["xyxy"] for i,j in pairs],[b[j]["xyxy"] for i,j in pairs],ref.GATES["final_box_atol_px"])
    score=error_stats([a[i]["score"] for i,j in pairs],[b[j]["score"] for i,j in pairs],ref.GATES["final_score_atol"])
    class_changes=[{"left":i,"right":j,"iou":ref.iou(a[i]["xyxy"],b[j]["xyxy"])} for i in ua for j in ub
                   if a[i]["class_id"]!=b[j]["class_id"] and ref.iou(a[i]["xyxy"],b[j]["xyxy"])>=.5]
    return {"matched":len(pairs),"pairs":pairs,"unmatched_left":ua,"unmatched_right":ub,"class_changes":class_changes,
            "box_errors_px":box,"score_errors":score,"candidate_changes":[[a[i].get("candidate_index"),b[j].get("candidate_index")] for i,j in pairs if a[i].get("candidate_index")!=b[j].get("candidate_index")],
            "pass":not ua and not ub and not class_changes and box["pass"] and score["pass"]}


def raw_comparison(a,b):
    ref.valid_raw(a); ref.valid_raw(b)
    return {"boxes_normalized":error_stats(a[:,:4]/1024,b[:,:4]/1024,1e-4,1e-4),
            "scores":error_stats(a[:,4:],b[:,4:],1e-4,1e-4)}


def sensitivity(a,b,da,db):
    sa,sb=a[0,4:].max(axis=0),b[0,4:].max(axis=0)
    active=(sa>.25)|(sb>.25)
    score_cross=np.flatnonzero((sa>.25)!=(sb>.25))
    cls_cross=np.flatnonzero(active & (a[0,4:].argmax(axis=0)!=b[0,4:].argmax(axis=0)))
    def near_nms(raw):
        x=raw[0].T; cls=x[:,4:].argmax(axis=1); score=x[:,4:].max(axis=1)
        ids=np.flatnonzero(score>.25)
        boxes=np.concatenate([x[:,:2]-x[:,2:4]/2,x[:,:2]+x[:,2:4]/2],axis=1)
        out=[]
        for k,i in enumerate(ids):
            for j in ids[k+1:]:
                if cls[i]==cls[j]:
                    v=ref.iou(boxes[i],boxes[j])
                    if abs(v-.7)<=1e-4: out.append([int(i),int(j),v])
        return out
    return {"confidence_crossings":score_cross.tolist(),"argmax_class_crossings":cls_cross.tolist(),
            "near_confidence_1e-4":np.flatnonzero((np.abs(sa-.25)<=1e-4)|(np.abs(sb-.25)<=1e-4)).tolist(),
            "near_nms_iou_1e-4_left":near_nms(a),"near_nms_iou_1e-4_right":near_nms(b),
            "kept_candidates_left":[d["candidate_index"] for d in da["detections"]],
            "kept_candidates_right":[d["candidate_index"] for d in db["detections"]]}


def controller_compare(case):
    paths=case["paths"]; base=paths["pytorch"]; result={}
    for name,path in paths.items():
        if name=="pytorch": continue
        ar={r["target_id"]:r for r in base["target_results"]}
        br={r["target_id"]:r for r in path["target_results"]}
        common=sorted(set(ar)&set(br)); changes=[]; timings=[]
        for key in common:
            x,y=ar[key],br[key]
            # Harness reports status and, on success, pulse fields at this level.
            if x.get("nozzle_index")!=y.get("nozzle_index") or x.get("reason")!=y.get("reason"):
                changes.append({"target_id":key,"left":x,"right":y})
            if all(k in x and k in y for k in ("arrival_time_us","on_time_us","off_time_us")):
                delta={k:y[k]-x[k] for k in ("arrival_time_us","on_time_us","off_time_us")}
                if any(delta.values()): timings.append({"target_id":key,**delta})
        def structure(p):
            return {"merged":[{k:v for k,v in row.items() if k not in ("on_time_us","off_time_us")} for row in p["merged_intervals"]],
                    "events":[{k:v for k,v in row.items() if k!="event_time_us"} for row in p["executed_events"]]}
        result[name]={"matched_targets":len(common),"unmatched_left":sorted(set(ar)-set(br)),"unmatched_right":sorted(set(br)-set(ar)),
                      "channel_or_rejection_changes":changes,"timestamp_differences_us":timings,
                      "max_abs_timestamp_difference_us":max([abs(d[k]) for d in timings for k in ("arrival_time_us","on_time_us","off_time_us")],default=0),
                      "merged_intervals_exact":base["merged_intervals"]==path["merged_intervals"],
                      "event_structure_equal":structure(base)==structure(path),
                      "events_exact":base["executed_events"]==path["executed_events"],
                      "final_states":path["final_states"],"final_states_equal":base["final_states"]==path["final_states"]}
    return result


def compare(args):
    import torch
    from ultralytics.data.augment import LetterBox
    manifest=json.loads(args.manifest.read_text())
    for key,current in [("gates",ref.GATES),("deployment",ref.DEPLOYMENT),("runtime",ref.RUNTIME),("controller_geometry",ref.GEOMETRY)]:
        if manifest[key]!=current: raise ValueError(f"frozen {key} changed")
    if ref.sha256(args.checkpoint)!=ref.CHECKPOINT_SHA256: raise ValueError("checkpoint mismatch")
    if args.output.exists(): raise ValueError("comparison output must be new")
    args.output.mkdir(parents=True)
    torch_model=ref.load_torch(args.checkpoint); ort=ref.session(args.model)
    rows=[]; controller_cases=[]
    for sample in manifest["samples"]:
        if ref.sha256(sample["path"])!=sample["sha256"]: raise ValueError("frozen sample changed")
        target=args.output/sample["id"]; target.mkdir()
        im=ref.decode(sample["path"]); tensor,g=ref.preprocess(im)
        native_im=LetterBox(new_shape=(1024,1024),auto=False,scale_fill=False,scaleup=True,center=True)(image=im)
        native_tensor=np.ascontiguousarray(native_im[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)
        if not np.array_equal(tensor,native_tensor): raise ValueError("Python preprocessing differs from pinned native LetterBox")
        tensor.tofile(target/"python.input.f32")
        raw_pt=ref.infer_torch(torch_model,tensor)
        raw_py=ort.run(None,{ort.get_inputs()[0].name:tensor})[0]
        raw_pt.tofile(target/"pytorch.raw.f32"); raw_py.tofile(target/"python.raw.f32")
        command=[str(args.executable.resolve()),"--model",str(args.model.resolve()),"--image",sample["path"]]
        for mode,extra in [("cpp",[]),("cpp_identical",["--tensor",str((target/"python.input.f32").resolve())])]:
            subprocess.run(command+["--output",str(target/(mode+".json")),"--dump-prefix",str(target/mode)]+extra,check=True,capture_output=True,text=True)
        cg=json.loads((target/"cpp.geometry.json").read_text())
        cpixels=np.fromfile(target/"cpp.bgr.u8",np.uint8).reshape(im.shape)
        cin=np.fromfile(target/"cpp.input.f32",np.float32).reshape(tensor.shape)
        cr=np.fromfile(target/"cpp.raw.f32",np.float32).reshape(raw_py.shape)
        cri=np.fromfile(target/"cpp_identical.raw.f32",np.float32).reshape(raw_py.shape)
        resized=(g["resized_width"],g["resized_height"])!=(g["original_width"],g["original_height"])
        A={"decoded_pixels_exact":bool(np.array_equal(im,cpixels)),"geometry_exact":all(g[k]==cg[k] for k in g),
           "geometry_python":g,"geometry_cpp":cg,
           "normalized_input":error_stats(tensor,cin,ref.GATES["input_resize_atol" if resized else "input_no_resize_atol"]),
           "changed_spatial_pixels":int(np.any(tensor!=cin,axis=1).sum()),"resized":resized,"pinned_LetterBox_exact":True}
        pt=ref.postprocess(raw_pt,g); py=ref.postprocess(raw_py,g); cpp=json.loads((target/"cpp.json").read_text())
        ptn=ref.native_postprocess(raw_pt,g); pyn=ref.native_postprocess(raw_py,g)
        for name,data in [("pytorch",pt),("onnx_python",py),("pytorch_native",ptn),("onnx_python_native",pyn)]:
            ref.write_json(target/(name+".json"),data)
        row={"id":sample["id"],"kind":sample["kind"],"A":A,"B":raw_comparison(raw_pt,raw_py),"C":raw_comparison(raw_py,cri),
             "raw_after_cpp_preprocessing":raw_comparison(raw_py,cr),
             "D":{"pytorch_vs_onnx_python":compare_detections(pt,py),"pytorch_vs_cpp":compare_detections(pt,cpp),"onnx_python_vs_cpp":compare_detections(py,cpp)},
             "pinned_native_postprocessing":{"pytorch":compare_detections(ptn,pt),"onnx_python":compare_detections(pyn,py)},
             "threshold_sensitivity":sensitivity(raw_pt,raw_py,pt,py)}
        row["pass"]=(A["decoded_pixels_exact"] and A["geometry_exact"] and A["normalized_input"]["pass"]
                     and all(v["pass"] for k in ("B","C") for v in row[k].values())
                     and all(v["pass"] for v in row["D"].values())
                     and all(v["pass"] for v in row["pinned_native_postprocessing"].values()))
        rows.append(row); ref.write_json(target/"comparison.json",row)
        if sample["kind"]=="real":
            paths={"pytorch":copy.deepcopy(pt),"onnx_python":copy.deepcopy(py),"onnx_cpp":copy.deepcopy(cpp)}
            for i,d in enumerate(paths["pytorch"]["detections"]): d["match_id"]=f"target-{i:04d}"
            for name in ("onnx_python","onnx_cpp"):
                pairs,ua,ub=match(pt["detections"],paths[name]["detections"])
                for i,j in pairs: paths[name]["detections"][j]["match_id"]=f"target-{i:04d}"
                for j in ub: paths[name]["detections"][j]["match_id"]=f"{name}-unmatched-{j:04d}"
            controller_cases.append({"id":sample["id"],"paths":paths})
        print(f'{sample["id"]}: {"PASS" if row["pass"] else "FAIL"}',flush=True)
    ref.write_json(args.output/"controller-input.json",{"cases":controller_cases})
    subprocess.run([str(args.controller.resolve()),"--input",str(args.output/"controller-input.json"),
                    "--output",str(args.output/"controller-output.json")],check=True)
    ctrl=json.loads((args.output/"controller-output.json").read_text())
    controller_results={case["id"]:controller_compare(case) for case in ctrl["cases"]}
    report={"manifest_sha256":ref.sha256(args.manifest),"model_sha256":ref.sha256(args.model),"runtime":ref.RUNTIME,
            "gates":ref.GATES,"samples":rows,"passed_samples":sum(r["pass"] for r in rows),"sample_count":len(rows),
            "controller":controller_results,"all_parity_gates_pass":all(r["pass"] for r in rows)}
    ref.write_json(args.output/"summary.json",report)
    print(json.dumps({"all_parity_gates_pass":report["all_parity_gates_pass"],"passed_samples":report["passed_samples"]}))
    return 0 if report["all_parity_gates_pass"] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="action",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--derived",type=Path,required=True); f.add_argument("--baseline-examples",type=Path,required=True); f.add_argument("--output",type=Path,required=True)
    c=sub.add_parser("compare")
    for name in ("manifest","checkpoint","model","executable","controller","output"): c.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.action=="freeze": freeze(args); return 0
    return compare(args)


if __name__=="__main__": raise SystemExit(main())
