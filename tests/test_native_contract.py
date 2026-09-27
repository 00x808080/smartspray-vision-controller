"""Optional compiled-CLI tests using tiny synthetic ONNX graphs, never private weights."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import json
import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EXECUTABLE = os.environ.get("SMARTSPRAY_INFER", "")


@unittest.skipUnless(EXECUTABLE and Path(EXECUTABLE).is_file(),
                     "NOT RUN: set SMARTSPRAY_INFER to the compiled native executable")
class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.image=self.root/"input.png"
        cv2.imwrite(str(self.image),np.zeros((17,33,3),np.uint8))

    def graph(self, input_shape=(1,3,1024,1024), output_shape=(1,6,10),
              names="{0: 'crop', 1: 'weed'}", task="detect", end2end="False",
              field=None, value=None, input_type=TensorProto.FLOAT):
        data=np.zeros(output_shape,np.float32)
        if field is not None: data.flat[field]=value
        node=helper.make_node("Constant",[],["output0"],value=numpy_helper.from_array(data))
        graph=helper.make_graph([node],"synthetic-contract",
             [helper.make_tensor_value_info("images",input_type,list(input_shape))],
             [helper.make_tensor_value_info("output0",TensorProto.FLOAT,list(output_shape))])
        # Synthetic graph generated directly in supported IR8; never modifies the trained graph.
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",18)],ir_version=8)
        helper.set_model_props(model,{"names":names,"task":task,"end2end":end2end})
        path=self.root/"model.onnx"; onnx.save(model,path)
        return path

    def run_cli(self, model=None, extra=(), output=None):
        return subprocess.run([EXECUTABLE,"--model",str(model or self.graph()),"--image",str(self.image),
                               "--output",str(output or self.root/"result.json"),*extra],
                              capture_output=True,text=True)

    def test_empty_success_deterministic(self):
        model=self.graph()
        self.assertEqual(self.run_cli(model).returncode,0)
        a=(self.root/"result.json").read_bytes()
        self.assertEqual(json.loads(a)["detections"],[])
        self.assertEqual(self.run_cli(model).returncode,0)
        self.assertEqual((self.root/"result.json").read_bytes(),a)

    def test_shape_type_rejections(self):
        for opts in [{"input_shape":(1,3,640,640)},{"input_shape":("batch",3,1024,1024)},
                     {"input_type":TensorProto.DOUBLE},{"output_shape":(1,7,10)},
                     {"output_shape":(1,10,6)},{"output_shape":(1,6,0)}]:
            with self.subTest(opts=opts):
                self.assertNotEqual(self.run_cli(self.graph(**opts)).returncode,0)

    def test_metadata_rejections(self):
        for opts in [{"names":"{0: 'weed', 1: 'crop'}"},{"names":"{}"},{"task":"segment"},{"end2end":"True"}]:
            with self.subTest(opts=opts):
                self.assertNotEqual(self.run_cli(self.graph(**opts)).returncode,0)

    def test_nonfinite_and_invalid_output(self):
        for field,value in [(0,np.nan),(1,np.inf),(20,-1),(40,1.1),(50,-.1)]:
            with self.subTest(field=field,value=value):
                self.assertNotEqual(self.run_cli(self.graph(field=field,value=value)).returncode,0)

    def test_invalid_model_and_tensor(self):
        model=self.root/"bad.onnx"; model.write_text("invalid")
        self.assertNotEqual(self.run_cli(model).returncode,0)
        tensor=self.root/"input.f32"; tensor.write_bytes(b"bad")
        self.assertNotEqual(self.run_cli(extra=["--tensor",str(tensor)]).returncode,0)

    def test_input_output_collision_preserves_image(self):
        before=self.image.read_bytes()
        self.assertNotEqual(self.run_cli(output=self.image).returncode,0)
        self.assertEqual(self.image.read_bytes(),before)

    def test_hardlink_input_output_collision(self):
        before=self.image.read_bytes()
        alias=self.root/"image-alias.json"; os.link(self.image,alias)
        self.assertNotEqual(self.run_cli(output=alias).returncode,0)
        self.assertEqual(self.image.read_bytes(),before)
        model=self.graph(); before=model.read_bytes()
        alias=self.root/"model-alias.json"; os.link(model,alias)
        self.assertNotEqual(self.run_cli(model,output=alias).returncode,0)
        self.assertEqual(model.read_bytes(),before)

    def test_hardlink_output_output_collision(self):
        first=self.root/"result.json"; first.write_text("preserve")
        alias=self.root/"timing.json"; os.link(first,alias)
        result=self.run_cli(extra=["--benchmark","30","--timing-output",str(alias)])
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(first.read_text(),"preserve")

    def test_controller_input_output_aliases(self):
        exe=Path(EXECUTABLE).parent/"controller_parity"
        source=self.root/"request.json"; source.write_text('{"cases": []}')
        for kind in ("same","symlink","hardlink"):
            dest=source if kind=="same" else self.root/(kind+".json")
            if kind=="symlink": dest.symlink_to(source)
            if kind=="hardlink": os.link(source,dest)
            result=subprocess.run([str(exe),"--input",str(source),"--output",str(dest)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(source.read_text(),'{"cases": []}')

    def test_actual_image_encoding_not_extension(self):
        cv2.imwrite(str(self.root/"a.jpg"),np.zeros((17,33,3),np.uint8))
        self.image.write_bytes((self.root/"a.jpg").read_bytes())
        self.assertNotEqual(self.run_cli().returncode,0)


if __name__=="__main__": unittest.main()
