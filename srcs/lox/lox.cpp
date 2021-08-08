//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/global_setting/global_setting.h"
#include "lox/parser.h"
#include "lox/scanner.h"
#include "lox/visitors/ast_printer/ast_printer.h"
#include "lox/visitors/evaluator/environment.h"
#include "lox/visitors/evaluator/evaluator.h"

namespace lox {
static bool g_debug = true;
Lox::Lox() {
  global_env_ = std::make_shared<Environment>();
  evaluator_ = std::make_shared<StmtEvaluator>(global_env_);
}

std::string Lox::CLIHelpString() { return std::string(); }

Error Lox::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  GlobalSetting().interactive_mode = false;
  return RunStream(&infile);
}

Error Lox::RunPrompt() {
  GlobalSetting().interactive_mode = true;
  return RunStream(&std::cin);
}

Error Lox::RunStream(std::istream *istream) {
  Error err;
  if (GlobalSetting().interactive_mode) {
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
