//
// LICENSE: MIT
//

#include "lox/lox.h"

#include <fstream>
#include <iostream>

namespace lox {
std::string Lox::CLIHelpString() { return std::__cxx11::string(); }

Error Lox::RunFile(const std::string &file_path) {
  std::ifstream infile(file_path);
  return RunStream(&infile, &std::cout, false);
}

Error Lox::RunPrompt() { return RunStream(&std::cin, &std::cout, true); }

Error Lox::Run(const std::string &code) {
  std::cout << "Echo: " << code << std::endl;
  return Error();
}

Error Lox::RunStream(std::istream *istream, std::ostream *ostream,
                     bool interactive_mode) {
  std::string line;
  Error err;
  if (interactive_mode) {
    *ostream << ">> ";
  }
  while (std::getline(*istream, line)) {
    err.Append(Run(line));
    if (interactive_mode) {
      *ostream << ">> ";
    }
  }
  return err;
}

int Error::TOErrCode() { return 0; }

void Error::Append(const Error &new_err) {}
}  // namespace lox
