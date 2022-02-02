

#include <fstream>
#include <string>

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/frontend/parser.h"
#include "lox/frontend/scanner.h"

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
  Parser parser(&scanner);
  auto root = parser.Parse();
  AstPrinter printer;
  for (auto& stmt : root->body) {
    std::cout << printer.Print(stmt.get());
  }
  std::cout << std::endl;
  return 0;
}
