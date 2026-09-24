"""Native M4 CLI integration with tiny generated ONNX fixtures, no private artifacts."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EXECUTABLE = os.environ.get("SMARTSPRAY_VISION_DEMO", "")


@unittest.skipUnless(EXECUTABLE and Path(EXECUTABLE).is_file(),
                     "NOT RUN: set SMARTSPRAY_VISION_DEMO to the compiled native demo")
class DemoCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.assertTrue(cv2.imwrite(str(self.image), np.zeros((1024, 1024, 3), np.uint8)))
        self.config = self.root / "config.json"
        self.profile = json.loads((Path(__file__).resolve().parents[1]/"configs/demo.json").read_text())
        self.config.write_text(json.dumps(self.profile))
        self.model = self.root / "model.onnx"
        self.output = self.root / "run"

    def graph(self, classes=(1,), ys=None, mutate=None, runtime_error=False):
        data = np.zeros((1, 6, 10), np.float32)
        for i, cls in enumerate(classes):
            data[0, :4, i] = [320+i*128, (ys or [512]*len(classes))[i], 20, 20]
            data[0, 4+cls, i] = .8
        if mutate:
            mutate(data)
        nodes = [helper.make_node("Constant", [], ["values"], value=numpy_helper.from_array(data))]
        if runtime_error:
            nodes += [
                helper.make_node("Constant", [], ["indices"], value=numpy_helper.from_array(np.array([4],np.int64))),
                helper.make_node("Gather", ["images","indices"], ["bad"], axis=1),
                helper.make_node("ReduceSum", ["bad"], ["sum"], keepdims=0),
                helper.make_node("Add", ["values","sum"], ["output0"])]
        else:
            nodes += [helper.make_node("Identity", ["values"], ["output0"])]
        graph = helper.make_graph(nodes, "m4-synthetic",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1,3,1024,1024])],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1,6,10])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",18)], ir_version=8)
        helper.set_model_props(model, {"names":"{0: 'crop', 1: 'weed'}","task":"detect","end2end":"False"})
        onnx.save(model, self.model)

    def invoke(self, output=None, model=None, image=None, config=None, **kwargs):
        return subprocess.run([EXECUTABLE, "--model", str(model or self.model),
            "--image", str(image or self.image), "--config", str(config or self.config),
            "--output", str(output or self.output)], capture_output=True, text=True, **kwargs)

    def success(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"run.json", "events.csv", "annotated.png", "timeline.png"})
        run = json.loads((self.output/"run.json").read_text())
        rows = list(csv.DictReader(io.StringIO((self.output/"events.csv").read_text())))
        events = [{"event_time_us":int(r["event_time_us"]), "nozzle_index":int(r["nozzle_index"]),
                   "command":r["command"]} for r in rows]
        self.assertEqual(events, run["action"]["executed_events"])
        self.assertEqual(events, run["action"]["schedule"])
        self.assertEqual(run["action"]["final_states"], ["OFF"]*8)
        self.assertEqual(run["image"]["sha256"], hashlib.sha256(self.image.read_bytes()).hexdigest())
        self.assertEqual(run["model"]["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertNotIn("wall_clock_ms", run)
        self.assertIn("wall_clock_ms", json.loads(result.stdout))
        return run

    def test_complete_repeat(self):
        self.graph(classes=(0,1))
        run = self.success()
        self.assertEqual(run["counts"]["targets"], 1)
        self.assertEqual(run["trace"][0]["status"], "not_selected")
        self.assertEqual(run["trace"][1]["detection_id"], "d1")
        self.assertEqual(run["trace"][1]["target_id"], "t1")
        again = self.root/"repeat"
        result = self.invoke(output=again)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("run.json","events.csv"):
            self.assertEqual((self.output/name).read_bytes(), (again/name).read_bytes())
        for name in ("annotated.png","timeline.png"):
            self.assertTrue(np.array_equal(cv2.imread(str(self.output/name)), cv2.imread(str(again/name))))

    def test_empty_success(self):
        self.graph(classes=())
        self.assertEqual(self.success()["counts"]["events"], 0)

    def test_crops_only_success(self):
        self.graph(classes=(0,0))
        r = self.success()
        self.assertEqual(r["counts"]["crops_not_selected"], 2)
        self.assertEqual(r["counts"]["rejected"], 0)
        self.assertEqual(r["counts"]["events"], 0)

    def test_all_rejected_success(self):
        self.graph(classes=(1,1), ys=[1000,1000])
        r = self.success()
        self.assertEqual(r["counts"]["rejected"], 2)
        self.assertEqual(r["counts"]["events"], 0)
        self.assertTrue(all(t["plan"]["reason"]=="TOO_LATE" for t in r["trace"]))

    def test_mixed(self):
        self.graph(classes=(1,1,0), ys=[512,1000,512])
        r = self.success()
        self.assertEqual([r["counts"][k] for k in ("accepted","rejected","crops_not_selected","events")], [1,1,1,2])

    def test_configuration_fails_before_pipeline(self):
        for update in ({"speed_mps":0},{"capture_time_us":2**63},{"pulse_duration_us":.5},
                       {"simulated_processing_delay_us":-1},{"extra":1},
                       {"capture_time_us":2**63-1, "simulated_processing_delay_us":1}):
            with self.subTest(update=update):
                self.config.write_text(json.dumps(self.profile | update))
                r = self.invoke(model=self.root/"absent.onnx")
                self.assertNotEqual(r.returncode, 0)
                self.assertFalse(self.output.exists())
                self.assertNotIn("Cannot open hash input",r.stderr)
                self.assertEqual(r.stdout,"")

    def test_invalid_configuration_document(self):
        self.graph()
        valid=json.dumps(self.profile)
        for text in (valid+"TRAILING_INVALID_JSON",valid+"{}",valid[:-1],""):
            with self.subTest(text=text[-40:]):
                self.config.write_text(text)
                result=self.invoke()
                self.assertNotEqual(result.returncode,0)
                self.assertIn("parse_error",result.stderr)
                self.assertEqual(result.stdout,"")
                self.assertFalse(self.output.exists())

    def test_invalid_model(self):
        self.model.write_bytes(b"invalid ONNX")
        before = self.model.read_bytes()
        result = self.invoke()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.model.read_bytes(),before)
        self.assertEqual(result.stdout,"")

    def test_invalid_image(self):
        self.graph()
        for data in (b"bad",b"\x89PNG\r\n\x1a\ninvalid"):
            self.image.write_bytes(data)
            self.assertNotEqual(self.invoke().returncode,0)
            self.assertFalse(self.output.exists())
            self.assertEqual(self.image.read_bytes(),data)
        cv2.imwrite(str(self.image),np.zeros((10,10,4),np.uint8))
        self.assertNotEqual(self.invoke().returncode,0)
        self.assertFalse(self.output.exists())

    def test_invalid_model_output(self):
        for index,value in ((0,float("nan")),(20,-1),(40,1.1)):
            with self.subTest(index=index):
                self.graph(mutate=lambda a:a.flat.__setitem__(index,value))
                result = self.invoke()
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(self.output.exists())
                self.assertEqual(result.stdout,"")

    def test_inference_runtime_failure(self):
        self.graph(runtime_error=True)
        result = self.invoke()
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Gather",result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(result.stdout,"")

    def test_existing_output_and_aliases(self):
        self.graph()
        self.success()
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        for destination in (self.output,self.image,self.model,self.config):
            result = self.invoke(output=destination)
            self.assertNotEqual(result.returncode,0)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})
        for name,target in (("symlink",self.output),("dangling",self.root/"missing")):
            alias=self.root/name; alias.symlink_to(target)
            self.assertNotEqual(self.invoke(output=alias).returncode,0)
            self.assertTrue(alias.is_symlink())
        alias=self.root/"hardlink"; os.link(self.image,alias)
        before=self.image.read_bytes()
        self.assertNotEqual(self.invoke(output=alias).returncode,0)
        self.assertEqual(self.image.read_bytes(),before)

    def test_output_parent_failure(self):
        self.graph()
        result=self.invoke(output=self.root/"missing"/"run")
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.root/"missing").exists())
        parent=self.root/"readonly"; parent.mkdir(); parent.chmod(0o500)
        try:
            result=self.invoke(output=parent/"run")
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((parent/"run").exists())
        finally:
            parent.chmod(0o700)

    def test_partial_write_cleanup(self):
        self.graph()
        def limit_output():
            signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE,(1024,1024))
        result=self.invoke(preexec_fn=limit_output)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("cannot write binary output",result.stderr)
        self.assertFalse(self.output.exists(), "Caught write failure left partial output")
        self.assertEqual(result.stdout,"")

    def test_strict_cli(self):
        self.graph()
        for extra in ([],["--model"],["--unknown","x"]):
            result=subprocess.run([EXECUTABLE,*extra],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
