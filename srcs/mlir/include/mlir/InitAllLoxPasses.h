//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_INCLUDE_INITALLPASSES_H
#define LOX_SRCS_MLIR_INCLUDE_INITALLPASSES_H

#include "mlir/Conversion/MixedLoxToLLVM/MixedLoxToLLVM.h"

#include <cstdlib>

namespace mlir::lox {

// for now, only this pass is registered
inline void registerAllPasses() { registerMixedLoxToLLVMPasses(); }

} // namespace mlir::lox

#endif // LOX_SRCS_MLIR_INCLUDE_INITALLPASSES_H
