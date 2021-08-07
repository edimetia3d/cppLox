//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/ast/ast_printer.h"
#include "lox/ast/environment.h"
#include "lox/ast/eval_visitor.h"
#include "lox/parser.h"
#include "lox/scanner.h"

namespace lox {
static bool g_debug = true;
Lox::Lox() {
  global_env_ = std::make_shared<Environment>();
  evaluator_ = std::make_shared<StmtEvaluator>(global_env_);
}

std::string Lox::CLIHelpString() { return std::string(); }

Error Lox::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  return RunStream(&infile, false);
}

Error Lox::RunPrompt() { return RunStream(&std::cin, true); }

Error Lox::RunStream(std::istream *istream, bool interactive_mode) {
  Error err;
  if (interactive_mode) {
    std::string one_line;
    std::cout << ">> ";
    while (std::getline(*istream, one_line)) {
      if (one_line == "exit()") {
        break;
      }
      Eval(one_line);
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)), std::istreambuf_iterator<char>());
    Eval(all_line);
  }

  return err;
}

void Lox::Eval(const std::string &code) {
  Scanner scanner(code);
  auto err = scanner.Scan();

  Parser parser(scanner.Tokens());
  auto statements = parser.Parse();

  std::string output;
  try {
    for (auto &stmt : statements) {
      if (g_debug) {
        static StmtPrinter printer;
        std::cout << "Debug Stmt: `" << printer.Print(stmt) << "`" << std::endl;
      }
      evaluator_->Eval(stmt);
    }
  } catch (RuntimeError &rt_err) {
    std::cout << rt_err.what() << std::endl;
  }
}
}  // namespace lox
