"""Offline numerical reference, matching and pinned postprocessing tests."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from tools import onnx_deploy as ref
from tools.verify_inference import compare_detections, error_stats, match, sensitivity


def raw(rows):
    result=np.zeros((1,6,max(10,len(rows))),np.float32)
    for i,(box,crop,weed) in enumerate(rows):
        result[0,:,i]=[*box,crop,weed]
    return result


class InferenceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.g=ref.preprocess(np.zeros((1024,1024,3),np.uint8))[1]

    def test_rgb_and_normalization(self):
        image=np.empty((1024,1024,3),np.uint8); image[:]=[0,127,255]
        t,g=ref.preprocess(image)
        self.assertEqual(t.dtype,np.float32); self.assertTrue(t.flags.c_contiguous)
        np.testing.assert_array_equal(t[0,:,0,0],np.array([1,127/255,0],np.float32))
        self.assertEqual(g["r"],1)

    def test_odd_centered_padding(self):
        t,g=ref.preprocess(np.zeros((701,1024,3),np.uint8))
        self.assertEqual((g["top"],g["bottom"]), (161,162))
        self.assertEqual(float(t[0,0,0,0]),float(np.float32(114)/255))
        self.assertEqual(float(t[0,0,161,0]),0)

    def test_python_round_ties(self):
        for h,expected in [(1333,666),(1335,668)]:
            _,g=ref.preprocess(np.zeros((h,2048,3),np.uint8))
            self.assertEqual(g["resized_height"],expected)
            self.assertEqual(g["r"],.5)

    def test_scaleup_and_fractional_restore(self):
        _,g=ref.preprocess(np.zeros((17,33,3),np.uint8))
        self.assertGreater(g["r"],1)
        box=[123.125,g["top"]+100.25,456.75,g["top"]+200.5]
        got=ref.restore(box,g)
        self.assertNotEqual(got[0],round(got[0]))
        self.assertAlmostEqual(got[0],box[0]/g["r"],places=5)

    def test_restore_clip(self):
        self.assertEqual(ref.restore([-1,-5,1025,1028],self.g),[0,0,1024,1024])

    def test_decode_png_bmp_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            image=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
            for ext in (".png",".bmp"):
                p=Path(temp)/("image"+ext); self.assertTrue(cv2.imwrite(str(p),image))
                np.testing.assert_array_equal(ref.decode(p),image)

    def test_invalid_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            for image in [np.zeros((4,4),np.uint8),np.zeros((4,4,4),np.uint8),np.zeros((4,4,3),np.uint16)]:
                p=Path(temp)/"unsupported.png"; cv2.imwrite(str(p),image)
                with self.assertRaises(ValueError): ref.decode(p)
            p=Path(temp)/"broken.png"; p.write_bytes(b"invalid")
            with self.assertRaises(ValueError): ref.decode(p)
            with self.assertRaises(ValueError): ref.decode(Path(temp)/"wrong.jpg")

    def test_invalid_preprocessing(self):
        for a in [None,np.zeros((0,2,3),np.uint8),np.zeros((2,2),np.uint8),np.zeros((2,2,3),np.float32)]:
            with self.assertRaises(ValueError): ref.preprocess(a)

    def test_confidence_strict(self):
        r=raw([([5,5,4,4],.25,0),([20,20,4,4],np.nextafter(np.float32(.25),np.float32(1)),0)])
        self.assertEqual([d["candidate_index"] for d in ref.postprocess(r,self.g)["detections"]],[1])

    def test_class_aware_and_single_label(self):
        r=raw([([50,50,20,20],.9,0),([50,50,20,20],0,.8),([80,80,20,20],.7,.7)])
        out=ref.postprocess(r,self.g)["detections"]
        self.assertEqual([d["class_id"] for d in out],[0,1,0])

    def test_nms_tie_by_candidate(self):
        r=raw([([50,50,20,20],.9,0),([50,50,20,20],.9,0)])
        self.assertEqual([d["candidate_index"] for d in ref.postprocess(r,self.g)["detections"]],[0])

    def test_iou_strict_boundary(self):
        # Two width-17, height-1 boxes, offset3 => intersection14 / union20.
        r=raw([([8.5,.5,17,1],.9,0),([11.5,.5,17,1],.8,0)])
        self.assertEqual(len(ref.postprocess(r,self.g)["detections"]),2)
        r[0,0,1]=11.499
        self.assertEqual(len(ref.postprocess(r,self.g)["detections"]),1)

    def test_empty_success_serialization(self):
        out=ref.postprocess(raw([]),self.g)
        self.assertEqual(out["detections"],[]); self.assertEqual(out["schema_version"],1)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"out.json"; ref.write_json(p,out); a=p.read_bytes()
            ref.write_json(p,out); self.assertEqual(a,p.read_bytes()); self.assertEqual(json.loads(a),out)

    def test_max_det(self):
        r=raw([([i*4,1,1,1],.9,0) for i in range(310)])
        self.assertEqual(len(ref.postprocess(r,self.g)["detections"]),300)

    def test_invalid_raw(self):
        for r in [np.zeros((1,84,10),np.float32),np.zeros((1,6,0),np.float32),np.zeros((1,6,10),np.float64)]:
            with self.assertRaises(ValueError): ref.valid_raw(r)
        for field,value in [(0,np.nan),(4,np.inf),(4,1.01),(3,-1),(5,-.01)]:
            r=raw([]); r[0,field,0]=value
            with self.assertRaises(ValueError): ref.postprocess(r,self.g)

    def test_pinned_native_nms_reference(self):
        r=raw([([50,50,20,20],.95,0),([51,51,20,20],.8,0),([50,50,20,20],0,.85),
               ([99,75,3,2],.4,.39),([300,200,4,5],.25,0)])
        native=ref.native_postprocess(r,self.g); custom=ref.postprocess(r,self.g)
        self.assertTrue(compare_detections(native,custom)["pass"])
        self.assertEqual({d["candidate_index"] for d in native["detections"]},{0,2,3})

    def test_matching_permutation_and_class_change(self):
        a=ref.postprocess(raw([([50,50,10,10],.9,0),([80,80,10,10],0,.8)]),self.g)
        b=json.loads(json.dumps(a)); b["detections"].reverse()
        c=compare_detections(a,b)
        self.assertTrue(c["pass"]); self.assertEqual(c["pairs"],[(0,1),(1,0)])
        b["detections"][0]["class_id"]=0
        c=compare_detections(a,b)
        self.assertFalse(c["pass"]); self.assertEqual(len(c["class_changes"]),1)

    def test_maximum_cardinality_matching(self):
        a=[{"class_id":0,"xyxy":[0,0,10,10]},{"class_id":0,"xyxy":[3,0,13,10]}]
        b=[{"class_id":0,"xyxy":[1,0,11,10]},{"class_id":0,"xyxy":[-2,0,8,10]}]
        pairs,ua,ub=match(a,b)
        self.assertEqual(len(pairs),2); self.assertEqual(ua,[]); self.assertEqual(ub,[])

    def test_budget_not_silently_widened(self):
        self.assertFalse(error_stats([1],[1.001],1e-4,1e-4)["pass"])
        self.assertTrue(error_stats([1],[1.0001],1e-4,1e-4)["pass"])
        with self.assertRaises(ValueError): error_stats([np.nan],[0],1e-4)

    def test_threshold_sensitivity_exposed(self):
        a=raw([([50,50,10,10],.250001,0)]); b=a.copy(); b[0,4,0]=.249999
        out=sensitivity(a,b,ref.postprocess(a,self.g),ref.postprocess(b,self.g))
        self.assertEqual(out["confidence_crossings"],[0])
        self.assertEqual(out["near_confidence_1e-4"],[0])


if __name__=="__main__": unittest.main()
