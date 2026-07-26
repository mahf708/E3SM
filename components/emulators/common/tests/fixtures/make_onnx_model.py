#!/usr/bin/env python3
"""Generate the tiny ONNX model used by the onnx-backend test.

Written by hand with `onnx.helper` rather than exported from a framework so
the test fixture needs only the small `onnx` package, and so the graph — and
therefore what the test asserts — is completely explicit.

The model has a dynamic batch ("ncol") dimension, two inputs and two outputs:

    dT    = 2 * T + 1          # [ncol, 3] float32
    total = T @ ones(3,1) + ps # [ncol, 1] float32

Usage: make_onnx_model.py <output.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NLEV = 3


def build_model() -> onnx.ModelProto:
    T = helper.make_tensor_value_info("T", TensorProto.FLOAT, ["ncol", NLEV])
    ps = helper.make_tensor_value_info("ps", TensorProto.FLOAT, ["ncol", 1])
    dT = helper.make_tensor_value_info("dT", TensorProto.FLOAT, ["ncol", NLEV])
    total = helper.make_tensor_value_info("total", TensorProto.FLOAT, ["ncol", 1])

    initializers = [
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "offset"),
        numpy_helper.from_array(np.ones((NLEV, 1), dtype=np.float32), "ones"),
    ]

    nodes = [
        helper.make_node("Mul", ["T", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "offset"], ["dT"]),
        helper.make_node("MatMul", ["T", "ones"], ["column_sum"]),
        helper.make_node("Add", ["column_sum", "ps"], ["total"]),
    ]

    graph = helper.make_graph(
        nodes,
        "e3sm_emulator_test_affine",
        inputs=[T, ps],
        outputs=[dT, total],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="e3sm"
    )
    # Keep the IR version conservative so older runtimes can load it too.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <output.onnx>", file=sys.stderr)
        return 2
    onnx.save(build_model(), argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
