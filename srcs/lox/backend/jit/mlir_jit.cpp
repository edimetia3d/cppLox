//
// LICENSE: MIT
//

#include "mlir_jit.h"

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/jit/translation/ast_to_mlir.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"
#include "lox/passes/pass_runner.h"
#include "lox/passes/semantic_check/semantic_check.h"
#include "mlir/Dialect/lox/Dialect.h"

namespace lox::jit {

void MLIRJIT::Run(Scanner& scanner) {
  auto parser = Parser::Make(GlobalSetting().parser, &scanner);
  auto root = parser->Parse();
  if (!root) {
    throw ParserError("Parse failed");
  }
#ifndef NDEBUG
  if (root && GlobalSetting().debug) {
    AstPrinter printer;
    std::cout << printer.Print(root.get()) << std::endl;
  }
#endif
  PassRunner pass_runner;
  pass_runner.SetPass({std::make_shared<SemanticCheck>()});
  pass_runner.Run(root.get());
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::lox::LoxDialect>();

  mlir::OwningModuleRef module = ConvertASTToMLIR(context, root.get());
  if (!module) {
    throw ParserError("Translation failed");
  }
  module->dump();
}
}  // namespace lox::MLIRJIT
