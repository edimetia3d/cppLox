
#include <iostream>

#include "lox/lox.h"

using namespace lox;

int main(int argn, char* argv[]) {
  Error ret;
  Lox interpreter;
  if (argn > 2) {
    std::cout << Lox::CLIHelpString() << std::endl;
  } else if (argn == 2) {
    ret = interpreter.RunFile(argv[1]);
  } else {
    ret = interpreter.RunPrompt();
  }
  return ret.ToErrCode();
}