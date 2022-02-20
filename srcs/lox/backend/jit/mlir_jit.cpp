//
// LICENSE: MIT
//

#include "mlir_jit.h"

#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/jit/translation/ast_to_mlir.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"
#include "lox/passes/pass_runner.h"
#include "lox/passes/semantic_check/semantic_check.h"
#include "mlir/Dialect/lox/Dialect.h"
#include "mlir/Dialect/lox/Passes.h"

namespace lox::jit {

class MLIRJITImpl {
 public:
  MLIRJITImpl();
  void Run(Scanner &scanner);
  void HandleMLIROpitons();
  std::unique_ptr<FunctionStmt> GetLoxAST(Scanner &scanner) const;
};

MLIRJIT::MLIRJIT() { impl_ = std::make_shared<MLIRJITImpl>(); }
void MLIRJIT::Run(Scanner &scanner) { impl_->Run(scanner); }

MLIRJITImpl::MLIRJITImpl() { HandleMLIROpitons(); }

void MLIRJITImpl::Run(Scanner &scanner) {
  std::unique_ptr<FunctionStmt> root = GetLoxAST(scanner);

  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::lox::LoxDialect>();

  mlir::OwningModuleRef module = ConvertASTToMLIR(context, root.get());
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

  if (mlir::failed(pm.run(*module))) throw LoxError("Optimization failed");
  module->dump();
}

std::unique_ptr<FunctionStmt> MLIRJITImpl::GetLoxAST(Scanner &scanner) const {
  auto parser = Parser::Make(GlobalSetting().parser, &scanner);
  auto root = parser->Parse();
  if (!root) {
    throw ParserError("Parse failed");
  }
  PassRunner pass_runner;
  pass_runner.SetPass({std::make_shared<SemanticCheck>()});
  pass_runner.Run(root.get());
  return root;
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

}  // namespace lox::jit
