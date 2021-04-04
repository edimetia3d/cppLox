//
// LICENSE: MIT
//

#include "lox/lox.h"

namespace lox {
std::string Lox::CLIHelpString() { return std::__cxx11::string(); }
Error Lox::RunFile(std::string file_path) { return Error(); }
Error Lox::RunPrompt() { return Error(); }
int Error::TOErrCode() { return 0; }
}  // namespace lox
