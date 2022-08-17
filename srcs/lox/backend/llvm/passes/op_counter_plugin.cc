#include "llvm/ADT/STLExtras.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "lox/backend/llvm/passes/op_counter.h"

using namespace llvm;

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "HaltAnalyzer", "v0.1", [](PassBuilder &PB) {
            using OptimizationLevel = typename PassBuilder::OptimizationLevel;
            PB.registerOptimizerLastEPCallback([](ModulePassManager &MPM, OptimizationLevel OL) {
              MPM.addPass(createModuleToFunctionPassAdaptor(lox::llvm_jit::HaltAnalyzer()));
            });
          }};
}