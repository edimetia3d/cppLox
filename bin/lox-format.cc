

#include <fstream>
#include <string>

#include "lox/frontend/parser.h"
#include "lox/frontend/scanner.h"
#include "lox/passes/ast_printer/ast_printer.h"

using namespace lox;

int main(int argn, char* argv[]) {
  if (argn != 2) {
    printf("Usage: lox-format path/to/file");
    return 0;
  }
  std::string file_path = argv[1];
  std::ifstream infile(file_path);
  std::string all_line((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
  Scanner scanner(all_line, file_path);
  auto parser = Parser::Make(ParserType::RECURSIVE_DESCENT, &scanner);
  auto module = parser->Parse();
  AstPrinter printer;
  for (auto& stmt : module->Statements()) {
    std::cout << printer.Print(stmt.get());
  }
  std::cout << std::endl;
  return 0;
}
