#ifndef LOX_INTERFACE_TD
#define LOX_INTERFACE_TD

include "mlir/IR/OpBase.td"

// todo: all shape inference code should be cleaned up to support more dtypes/inferShapes
// currently we only support static shaped tensor and unranked tensor.
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.

    Note:
        This interface is only used by the shape inference pass, and there is some
        internal contract between the pass and the interface, e.g. the pass will
        always make sure that all operands have been inferred before calling into
        the interface, and all operands are guaranteed to be tensor types, so the
        interface can assume that without checking.
  }];

  let cppNamespace = "mlir::lox";
  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif
