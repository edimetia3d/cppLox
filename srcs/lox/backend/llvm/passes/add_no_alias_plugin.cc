

#include "llvm/ADT/STLExtras.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "lox/backend/llvm/passes/add_no_alias.h"

using namespace llvm;
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  auto register_cb = [](PassBuilder &PB) {
    // Use opt's `--passes` textual pipeline description to trigger
    // AddNoAlias
    using PipelineElement = typename PassBuilder::PipelineElement;
    PB.registerPipelineParsingCallback([](StringRef Name, FunctionPassManager &FPM, ArrayRef<PipelineElement>) {
      if (Name == "add_no_alias") {
        FPM.addPass(lox::llvm_jit::AddNoAlias());
        return true;
      }
      return false;
    });
  };
  return {LLVM_PLUGIN_API_VERSION, "AddNoAlias", "v0.1", register_cb};
}