//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/tree_walker.h"

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/tree_walker/evaluator/evaluator.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"
#include "lox/passes/pass_runner.h"
#include "lox/passes/semantic_check/semantic_check.h"

namespace lox::twalker {
TreeWalker::TreeWalker() { evaluator_ = std::make_shared<Evaluator>(); }
void TreeWalker::Run(Scanner &scanner) {
  Parser parser(&scanner);
  auto root = parser.Parse();
#ifndef NDEBUG
  if (root && GlobalSetting().debug) {
    AstPrinter printer;
    std::cout << printer.Print(root.get()) << std::endl;
  }
#endif
  PassRunner pass_runner;
  pass_runner.SetPass({std::make_shared<SemanticCheck>()});
  pass_runner.Run(root.get());
  evaluator_->LaunchScript(root.get());
}

}  // namespace lox
