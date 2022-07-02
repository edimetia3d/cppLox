//
// LICENSE: MIT
//

#include "mlir_jit.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/mlir/translation/ast_to_mlir.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"
#include "lox/passes/pass_runner.h"
#include "lox/passes/semantic_check/semantic_check.h"
#include "mlir/Dialect/lox/Dialect.h"
#include "mlir/Dialect/lox/Passes.h"

namespace lox::mlir_jit {

class MLIRJITImpl : public BackEnd {
 public:
  MLIRJITImpl();
  void Run(Scanner &scanner) override;
  void HandleMLIROpitons();
};

MLIRJIT::MLIRJIT() { impl_ = std::make_shared<MLIRJITImpl>(); }
void MLIRJIT::Run(Scanner &scanner) { impl_->Run(scanner); }

MLIRJITImpl::MLIRJITImpl() { HandleMLIROpitons(); }

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/GlobalSetting().opt_level ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/GlobalSetting().opt_level ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

void MLIRJITImpl::Run(Scanner &scanner) {
  auto lox_module = BuildASTModule(scanner);

  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::lox::LoxDialect>();

  mlir::OwningModuleRef module = ConvertASTToMLIR(context, lox_module.get());
  if (!module) {
    throw ParserError("Translation failed");
  }
  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(mlir::createInlinerPass());  // always inline to support infer-shape
  // Now that there is only one function, we can infer the shapes of each of
  // the operations.
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(mlir::lox::createShapeInferencePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());

  // Partially lower the toy dialect with a few cleanups afterwards.
  optPM.addPass(mlir::lox::createLowerToAffinePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());

  if (GlobalSetting().opt_level) {
    // Add optimizations if enabled.
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createAffineScalarReplacementPass());
  }

  pm.addPass(mlir::lox::createLowerToLLVMPass());

  if (mlir::failed(pm.run(*module))) throw LoxError("Optimization failed");
  module->dump();
  std::cout << " ============== LLVM IR Dump =================" << std::endl;
  dumpLLVMIR(*module);
  std::cout << " ============== JIT Run =================" << std::endl;
  runJit(*module);
}

void MLIRJITImpl::HandleMLIROpitons() {
  std::string raw_args = GlobalSetting().mlir_cli_options;
  std::vector<const char *> args;
  args.push_back("lox");
  if (GlobalSetting().debug) {
    args.push_back("-mlir-print-debuginfo");
  }
  auto iter = raw_args.begin();
  while (iter != raw_args.end() && *iter == ' ') {
    ++iter;
  }
  if (iter != raw_args.end()) {
    args.push_back(&*iter);
  }
  bool new_blank = false;
  while (iter != raw_args.end()) {
    if (*iter == ' ') {
      new_blank = true;
      *iter = '\0';
    } else {
      if (new_blank) {
        args.push_back(&*iter);
      }
      new_blank = false;
    }
    ++iter;
  }

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(args.size(), args.data(), "Lox MLIR JIT");
}

}  // namespace lox::mlir_jit
