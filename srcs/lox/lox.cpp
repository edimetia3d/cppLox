//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/parser.h"
#include "lox/scanner.h"

namespace lox {
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
      std::string oneline_output;
      auto line_err = Eval(one_line, &oneline_output);
      if (line_err.ToErrCode() > 0) {
        std::cerr << line_err.Message() << std::endl;
      }
      std::cout << oneline_output << std::endl;
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)),
                         std::istreambuf_iterator<char>());
    err = Eval(all_line, nullptr);
  }

  return err;
}

Error Lox::Eval(const std::string &code, std::string *eval_output) {
  Scanner scanner(code);
  std::string output;
  auto err = scanner.Scan();

  Parser parser(scanner.Tokens());
  auto expr = parser.Parse();
  if (eval_output) {
    *eval_output = std::move(output);
  }
  return err;
}
}  // namespace lox
