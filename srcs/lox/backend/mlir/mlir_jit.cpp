//
// LICENSE: MIT
//

#include "mlir_jit.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "lox/backend/mlir/translation/ast_to_mlir.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"
#include "lox/passes/ast_printer/ast_printer.h"
#include "mlir/Conversion/LoxToMixedLox/LoxToMixedLox.h"
#include "mlir/Conversion/MixedLoxToLLVM/MixedLoxToLLVM.h"
#include "mlir/Dialect/Lox/IR/LoxDialect.h"
#include "mlir/Dialect/Lox/Transforms/Passes.h"

#include "mlir/InitAllLoxDialects.h"

namespace lox::mlir_jit {

class MLIRJITImpl : public BackEnd {
public:
  MLIRJITImpl();
  void Run(Scanner &scanner) override;
  void HandleMLIROpitons();
  mlir::MLIRContext context_;
};

MLIRJIT::MLIRJIT() { impl_ = std::make_shared<MLIRJITImpl>(); }
void MLIRJIT::Run(Scanner &scanner) { impl_->Run(scanner); }

MLIRJITImpl::MLIRJITImpl() {
  HandleMLIROpitons();
  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  mlir::lox::registerAllDialects(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  context_.appendDialectRegistry(registry);
  context_.loadAllAvailableDialects();
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
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

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), tmOrError.get().get());

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
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/GlobalSetting().opt_level ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
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

  mlir::OwningOpRef<mlir::ModuleOp> module = ConvertASTToMLIR(context_, lox_module.get());
  if (!module) {
    throw ParserError("Translation failed");
  }
  mlir::PassManager pm(&context_);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(mlir::createInlinerPass()); // always inline to support infer-shape
  // Now that there is only one function, we can infer the shapes of each of
  // the operations.
  {
    mlir::OpPassManager &optPM = pm.nest<mlir::lox::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::lox::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }
  // Partially lower the toy dialect with a few cleanups afterwards.
  pm.addPass(mlir::lox::createLowerLoxToMixedLoxPass());

  {
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    if (GlobalSetting().opt_level) {
      // Add optimizations if enabled.
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }
  pm.addPass(mlir::lox::createLowerMixedLoxToLLVMPass());

  // todo: Upstream added some debug info? check it later
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

  if (mlir::failed(pm.run(*module)))
    throw LoxError("Optimization failed");
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

} // namespace lox::mlir_jit
