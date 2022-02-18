//
// License: MIT
//

#include "ast_to_mlir.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <functional>

#include "lox/ast/ast.h"
#include "lox/common/finally.h"
#include "lox/common/lox_error.h"
#include "mlir/Dialect/lox/Dialect.h"

using namespace mlir::lox;
using namespace lox;
using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

class MLIRTranslationError : public LoxError {
 public:
  using LoxError::LoxError;
};

namespace {
struct TensorType {
  std::vector<int64_t> shape;
};

struct TensorLiteral {
  TensorLiteral(TensorExpr *tensor) {
    auto shape_expr = tensor->shape->DynAs<ListExpr>()->comma_expr->DynAs<CommaExpr>();
    std::vector<int64_t> shape;
    int element_num = 0;
    for (auto &expr : shape_expr->elements) {
      auto int_expr = std::stoi(expr->DynAs<LiteralExpr>()->attr->value->lexeme);
      if (element_num == 0) {
        element_num = int_expr;
      } else {
        element_num *= int_expr;
      }
      shape.push_back(int_expr);
    }
    data.resize(element_num);
    auto data_expr = tensor->data->DynAs<ListExpr>()->comma_expr->DynAs<CommaExpr>();
    int index = 0;
    for (auto &expr : data_expr->elements) {
      auto float_expr = std::stod(expr->DynAs<LiteralExpr>()->attr->value->lexeme);
      data[index] = float_expr;
      index++;
    }
    type = TensorType{.shape = shape};
  }
  TensorType type;
  std::vector<double> data;
};

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file;  ///< filename.
  int line;                           ///< line number.
  int col;                            ///< column number.
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::string> params;  // variable nodes

 public:
  PrototypeAST(FunctionStmt *fn) {
    location = Location{
        .file = std::make_shared<std::string>(fn->attr->name->file_name),
        .line = fn->attr->name->line,
        .col = fn->attr->name->col,
    };
    name = fn->attr->name->lexeme;
    if (fn->comma_expr_params) {
      for (auto &expr : fn->comma_expr_params->As<CommaExpr>()->elements) {
        params.push_back(expr->DynAs<VariableExpr>()->attr->name->lexeme);
      }
    }
  }

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::string> getArgs() { return params; }
};

class ASTToMLIR : public lox::AstNodeVisitor<mlir::Value> {
 public:
  ASTToMLIR(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp Convert(FunctionStmt *root_func) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    for (auto &stmt : root_func->body) {
      stmt->Accept(this);
    }
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      throw MLIRTranslationError("module verification error");
    }
    return theModule;
  }

 protected:
  void Visit(LogicalExpr *node) override { VisitorReturn(nullptr); }
  void Visit(BinaryExpr *node) override {
    node->left->Accept(this);
    mlir::Value lhs = PopRet();
    if (!lhs) throw MLIRTranslationError("error in left side of binary expr");
    node->right->Accept(this);
    mlir::Value rhs = PopRet();
    if (!rhs) throw MLIRTranslationError("error in right side of binary expr");
    auto location = loc(node->attr->op);

    // Derive the operation name from the binary operator. At the moment we only
    // support '+'
    switch (node->attr->op->lexeme[0]) {
      case '+':
        VisitorReturn(builder.create<AddOp>(location, lhs, rhs));
    }
    throw MLIRTranslationError("invalid binary operator '");
  }
  void Visit(GroupingExpr *node) override { VisitorReturn(nullptr); }
  void Visit(LiteralExpr *node) override {
    // only number are supported for now
    // all number will be trated as a zero-rank tensor
    assert(node->attr->value->type == TokenType::NUMBER);
    auto ret = builder.create<TensorOp>(loc(node->attr->value), std::stod(node->attr->value->lexeme));
    VisitorReturn(ret);
  }
  void Visit(UnaryExpr *node) override { VisitorReturn(nullptr); }
  void Visit(VariableExpr *node) override {
    if (auto variable = symbolTable.lookup(node->attr->name->lexeme)) VisitorReturn(variable);

    throw MLIRTranslationError("error: unknown variable '");
  }
  void Visit(AssignExpr *node) override { VisitorReturn(nullptr); }
  void Visit(CallExpr *node) override {
    llvm::StringRef callee = node->callee->DynAs<VariableExpr>()->attr->name->lexeme;
    auto location = loc(node->attr->src_token);

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : node->comma_expr_args->As<CommaExpr>()->elements) {
      expr->Accept(this);
      auto arg = PopRet();
      if (!arg) throw MLIRTranslationError("error in argument of call expr");
      operands.push_back(arg);
    }
    if (callee == "transpose") {
      VisitorReturn(builder.create<TransposeOp>(location, operands[0]));
    }
    VisitorReturn(builder.create<GenericCallOp>(location, callee, operands));
  }
  void Visit(GetAttrExpr *node) override { VisitorReturn(nullptr); }
  void Visit(SetAttrExpr *node) override { VisitorReturn(nullptr); }
  void Visit(PrintStmt *node) override {
    node->expression->Accept(this);
    auto arg = PopRet();
    if (!arg) throw MLIRTranslationError("error in print expr");

    builder.create<PrintOp>(loc(node->attr->src_token), arg);
  }
  void Visit(ReturnStmt *node) override {
    auto location = loc(node->attr->src_token);

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (node->value) {
      node->value->Accept(this);
      expr = PopRet();
      if (!(expr)) throw MLIRTranslationError("error in return expression");
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
  }
  void Visit(WhileStmt *node) override {}
  void Visit(ForStmt *node) override {}
  void Visit(BreakStmt *node) override {}
  void Visit(ExprStmt *node) override {}
  void Visit(VarDeclStmt *node) override {
    node->initializer->Accept(this);
    mlir::Value value = PopRet();
    if (!value) throw MLIRTranslationError("error in initializer of variable");

    // Register the value in the symbol table.
    if (failed(declare(node->attr->name->lexeme, value))) throw MLIRTranslationError("error in variable declaration");
  }

  mlir::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());
    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(), getType(TensorType{}));
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    return mlir::FuncOp::create(location, proto.getName(), func_type);
  }

  void Visit(FunctionStmt *node) override {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);
    PrototypeAST fn_proto(node);
    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(mlirGen(fn_proto));
    bool something_wrong = true;
    Finally function_guard([&] {
      if (something_wrong) {
        function.erase();
      }
    });
    if (!function) throw MLIRTranslationError("function creation error");

    auto &entryBlock = *function.addEntryBlock();
    auto protoArgs = fn_proto.getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(llvm::StringRef(std::get<0>(nameValue)), std::get<1>(nameValue))))
        throw MLIRTranslationError("variable declaration error");
    }

    // Set the insertion point in the builder to the beginning of the function
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    for (auto &stmt : node->body) {
      stmt->Accept(this);
    }

    // emit default return
    ReturnOp returnOp;
    if (!entryBlock.empty()) returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(fn_proto.loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(function.getType().getInputs(), getType(TensorType{})));
    }

    theModule.push_back(function);
    something_wrong = false;
  }
  void Visit(ClassStmt *node) override {}
  void Visit(BlockStmt *node) override {
    ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
    for (auto &stmt : node->statements) {
      stmt->Accept(this);
    }
  }
  void Visit(IfStmt *node) override {}
  void Visit(CommaExpr *node) override {}
  void Visit(ListExpr *node) override {}
  void Visit(GetItemExpr *node) override {}
  void Visit(TensorExpr *node) override {
    auto tensor_literal = TensorLiteral(node);
    auto type = getType(tensor_literal.type);
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(tensor_literal.type.shape, elementType);
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(tensor_literal.data));
    auto ret = builder.create<TensorOp>(loc(node->attr->src_token), type, dataAttribute);
    VisitorReturn(ret);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var)) return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return mlir::FileLineColLoc::get(builder.getIdentifier(*loc.file), loc.line, loc.col);
  }

  mlir::Location loc(Token token) {
    return loc(Location{std::make_shared<std::string>(token->file_name), token->line, token->col});
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty()) return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const TensorType &type) { return getType(type.shape); }

  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
};

}  // namespace

namespace lox::jit {

// The public API for codegen.
mlir::OwningModuleRef ConvertASTToMLIR(mlir::MLIRContext &context, lox::ASTNode *root) {
  return ASTToMLIR(context).Convert(root->DynAs<FunctionStmt>());
}

}  // namespace lox::jit
