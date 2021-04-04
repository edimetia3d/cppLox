//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

#include "lox/scanner.h"

namespace lox {
std::string Lox::CLIHelpString() { return std::__cxx11::string(); }

Error Lox::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  return RunStream(&infile, false);
}

Error Lox::RunPrompt() { return RunStream(&std::cin, true); }

Error Lox::Run(const std::string &code) {
  std::string eval_output;
  Eval(code, &eval_output);
  std::cout << eval_output << std::endl;
  return Error();
}

Error Lox::RunStream(std::istream *istream, bool interactive_mode) {
  Error err;
  if (interactive_mode) {
    std::string one_line;
    std::cout << ">> ";
    while (std::getline(*istream, one_line)) {
      err.Append(Run(one_line));
      std::cout << ">> ";
    }
  } else {
    std::string all_line((std::istreambuf_iterator<char>(*istream)),
                         std::istreambuf_iterator<char>());
    err = Run(all_line);
  }

  return err;
}
Error Lox::Eval(const std::string &code, std::string *eval_output) {
  Scanner scanner(code);
  std::string ret;
  auto tokens = scanner.Scan();
  for (auto &token : tokens) {
    ret += token.Str();
  }
  *eval_output = std::move(ret);
  return Error();
}

int Error::TOErrCode() { return 0; }

void Error::Append(const Error &new_err) {}
}  // namespace lox
