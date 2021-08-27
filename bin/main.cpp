
#include <iostream>

#include "lox/lox.h"
#include "lox/version.h"

using namespace lox;

void PrintBuildInfo() {
  auto str = std::string("Build info:\nSHA1: ") + std::string(version::GitSha1()) +
             "\nDate: " + std::string(version::GitDate());
  std::cout << str << std::endl;
}

int main(int argn, char* argv[]) {
  PrintBuildInfo();
  InterpreterError ret = lox::InterpreterError::NO_ERROR;
  LoxInterpreter interpreter;
  if (argn > 2) {
    std::cout << LoxInterpreter::CLIHelpString() << std::endl;
  } else if (argn == 2) {
    ret = interpreter.RunFile(argv[1]);
  } else {
    ret = interpreter.RunPrompt();
  }
  return static_cast<int>(ret);
}
