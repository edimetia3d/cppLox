//
// License: MIT
//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/InitAllLoxDialects.h"
#include "mlir/InitAllLoxPasses.h"

using namespace llvm;
using namespace mlir;
int main(int argc, char **argv) {
  registerAllPasses();
  lox::registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  lox::registerAllDialects(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry));
}