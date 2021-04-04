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
  return RunStream(&infile);
}

Error Lox::RunPrompt() { return RunStream(&std::cin); }

Error Lox::Run(const std::string &code) {
  std::cout << "Echo: " << code << std::endl;
  return Error();
}

Error Lox::RunStream(std::istream *stream) {
  std::string line;
  Error err;
  while (std::getline(*stream, line)) {
    err.Append(Run(line));
  }
  return err;
}

int Error::TOErrCode() { return 0; }

void Error::Append(const Error &new_err) {}
}  // namespace lox
