//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/tree_walker.h"

#include "lox/backend/tree_walker/evaluator/evaluator.h"

namespace lox::twalker {
TreeWalker::TreeWalker() { evaluator_ = std::make_shared<Evaluator>(); }
void TreeWalker::Run(Scanner &scanner) {
  std::unique_ptr<FunctionStmt> root = BuildAST(scanner);
  evaluator_->LaunchScript(root.get());
}

}  // namespace lox
