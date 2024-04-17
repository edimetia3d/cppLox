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
#include "mlir/Dialect/Lox/IR/LoxDialect.h"

using namespace mlir::lox;
using namespace lox;
using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

class MLIRTranslationError : public LoxError {
public:
  using LoxError::LoxError;
};

namespace {

struct TensorLiteral {
  TensorLiteral(TensorExpr *tensor) {
    auto shape_expr = tensor->shape->DynAs<ListExpr>()->comma_expr->DynAs<CommaExpr>();
    int element_num = 0;
    for (auto &expr : shape_expr->elements) {
      auto int_expr = std::stoi(expr->DynAs<LiteralExpr>()->attr->value->lexeme);
      if (element_num == 0) {
        element_num = int_expr;
      } else {
        element_num *= int_expr;
      }
      shape_.push_back(int_expr);
    }
    shape = shape_;
    data.resize(element_num);
    auto data_expr = tensor->data->DynAs<ListExpr>()->comma_expr->DynAs<CommaExpr>();
    int index = 0;
    for (auto &expr : data_expr->elements) {
      auto float_expr = std::stod(expr->DynAs<LiteralExpr>()->attr->value->lexeme);
      data[index] = float_expr;
      index++;
    }
  }
  ArrayRef<int64_t> shape;
  std::vector<double> data;

private:
  std::vector<int64_t> shape_;
};

class FunctionInfo {
  std::string name;
  std::vector<std::string> params; // variable nodes

public:
  Token location;
  FunctionInfo(FunctionStmt *fn) {
    name = fn->attr->name->lexeme;
    location = fn->attr->name;
    if (fn->comma_expr_params) {
      for (auto &expr : fn->comma_expr_params->As<CommaExpr>()->elements) {
        params.push_back(expr->DynAs<VariableExpr>()->attr->name->lexeme);
      }
    }
  }

  llvm::StringRef FnName() const { return name; }
  llvm::ArrayRef<std::string> ArgNames() { return params; }
};

class ASTToMLIR : public lox::ASTNodeVisitor<mlir::Value> {
public:
  ASTToMLIR(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp Convert(std::vector<StmtPtr> &global_stmts) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    for (auto &stmt : global_stmts) {
      NoValueVisit(stmt);
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
    mlir::Value lhs = ValueVisit(node->left);
    if (!lhs)
      throw MLIRTranslationError("error in left side of binary expr");
    mlir::Value rhs = ValueVisit(node->right);
    if (!rhs)
      throw MLIRTranslationError("error in right side of binary expr");
    auto location = Loc(node->attr->op);

    // Derive the operation name from the binary operator.
    switch (node->attr->op->lexeme.Data()[0]) {
    case '+':
      VisitorReturn(builder.create<AddOp>(location, lhs, rhs));
    case '*':
      VisitorReturn(builder.create<MulOp>(location, lhs, rhs));
    }
    throw MLIRTranslationError("invalid binary operator '");
  }
  void Visit(GroupingExpr *node) override { VisitorReturn(nullptr); }
  void Visit(LiteralExpr *node) override {
    // only number are supported for now
    // all number will be trated as a zero-rank tensor
    assert(node->attr->value->type == TokenType::NUMBER);
    auto ret = builder.create<ConstantOp>(Loc(node->attr->value), std::stod(node->attr->value->lexeme));
    VisitorReturn(ret);
  }
  void Visit(UnaryExpr *node) override { VisitorReturn(nullptr); }
  void Visit(VariableExpr *node) override {
    if (auto variable = symbolTable.lookup(node->attr->name->lexeme.Str()))
      VisitorReturn(variable);

    throw MLIRTranslationError("error: unknown variable '");
  }
  void Visit(AssignExpr *node) override { VisitorReturn(nullptr); }
  void Visit(CallExpr *node) override {
    llvm::StringRef callee = node->callee->DynAs<VariableExpr>()->attr->name->lexeme.Str();
    auto location = Loc(node->attr->src_token);

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    if (node->comma_expr_args) {
      for (auto &expr : node->comma_expr_args->As<CommaExpr>()->elements) {
        auto arg = ValueVisit(expr);
        if (!arg)
          throw MLIRTranslationError("error in argument of call expr");
        operands.push_back(arg);
      }
    }
    // try built in function first
    if (callee == "transpose") {
      if (!node->comma_expr_args || node->comma_expr_args->As<CommaExpr>()->elements.size() != 1) {
        throw MLIRTranslationError("built in function transpose takes one argument");
      }
      VisitorReturn(builder.create<TransposeOp>(location, operands[0]));
    }
    auto calledFunc = functionMap.lookup(callee);
    auto call_op = builder.create<GenericCallOp>(location, calledFunc.getFunctionType().getResult(0),
                                                 mlir::SymbolRefAttr::get(builder.getContext(), callee), operands);
    VisitorReturn(call_op);
  }
  void Visit(GetAttrExpr *node) override { VisitorReturn(nullptr); }
  void Visit(SetAttrExpr *node) override { VisitorReturn(nullptr); }
  void Visit(PrintStmt *node) override {
    auto arg = ValueVisit(node->expression);
    if (!arg)
      throw MLIRTranslationError("error in print expr");

    builder.create<PrintOp>(Loc(node->attr->src_token), arg);
  }
  void Visit(ReturnStmt *node) override {
    auto location = Loc(node->attr->src_token);

    // 'return' takes an optional expression, handle that case here.
    mlir::Value ret_v = nullptr;
    if (node->value) {
      ret_v = ValueVisit(node->value);
      if (!(ret_v)) throw MLIRTranslationError("error in return expression");
    }
    builder.create<ReturnOp>(location, ret_v);
  }
  void Visit(WhileStmt *node) override {}
  void Visit(ForStmt *node) override {}
  void Visit(BreakStmt *node) override {}
  void Visit(ExprStmt *node) override {}
  void Visit(VarDeclStmt *node) override {
    if (!node->initializer) {
      throw MLIRTranslationError("error: variable declaration must have initializer");
    }
    mlir::Value value = ValueVisit(node->initializer);
    if (!value)
      throw MLIRTranslationError("error in initializer of variable");

    // Register the value in the symbol table.
    if (failed(DeclareNamedValue(node->attr->name->lexeme.Str(), value)))
      throw MLIRTranslationError("error in variable declaration");
  }

  void Visit(FunctionStmt *node) override {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);
    FunctionInfo fn_info(node);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    auto fn_location = Loc(fn_info.location);
    llvm::SmallVector<mlir::Type, 4> arg_types;
    for (auto arg : fn_info.ArgNames()) {
      arg_types.push_back(TensorType({}));
    }

    // return llvm::none by default , we could update it later
    auto function = builder.create<mlir::lox::FuncOp>(fn_location, fn_info.FnName(),
                                                      builder.getFunctionType(arg_types, builder.getNoneType()));

    if (fn_info.FnName()[0] == '_') {
      function.setPrivate(); // Private functions could be inlined then removed by inline pass
    }

    bool something_wrong = true;
    Finally function_guard([&] {
      if (something_wrong) {
        function.erase();
      }
    });
    if (!function)
      throw MLIRTranslationError("function creation error");

    auto &entry_block = function.front();

    // Declare all the function arguments in the symbol table.
    // function argument will live in it's entry block
    for (const auto pair : llvm::zip(fn_info.ArgNames(), entry_block.getArguments())) {
      auto name = llvm::StringRef(std::get<0>(pair));
      auto value = std::get<1>(pair);
      if (failed(DeclareNamedValue(name, value)))
        throw MLIRTranslationError("variable declaration error");
    }

    // Set the insertion point in the builder to the beginning of the function
    builder.setInsertionPointToStart(&entry_block);

    // Emit the body of the function.
    for (auto &stmt : node->body) {
      NoValueVisit(stmt);
    }

    // try get last return op from function body
    ReturnOp returnOp;
    if (!entry_block.empty()) {
      returnOp = dyn_cast<ReturnOp>(entry_block.back());
      if (returnOp && returnOp.hasOperand()) {
        // update the return type of function
        function.setType(builder.getFunctionType(arg_types, TensorType({})));
      }
    }

    // emit default return op if last op is not return op
    if (!returnOp) {
      builder.create<ReturnOp>(fn_location);
    }

    functionMap.insert({fn_info.FnName(), function});
    something_wrong = false;
  }
  void Visit(ClassStmt *node) override {}
  void Visit(BlockStmt *node) override {
    ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
    for (auto &stmt : node->statements) {
      NoValueVisit(stmt);
    }
  }
  void Visit(IfStmt *node) override {}
  void Visit(CommaExpr *node) override {}
  void Visit(ListExpr *node) override {}
  void Visit(GetItemExpr *node) override {}
  void Visit(TensorExpr *node) override {
    auto tensor_literal = TensorLiteral(node); // all tensor var will have to be constant for now
    auto type = TensorType(tensor_literal.shape);
    auto dataAttribute = mlir::DenseElementsAttr::get(type, llvm::ArrayRef(tensor_literal.data));
    auto ret = builder.create<ConstantOp>(Loc(node->attr->src_token), type, dataAttribute);
    VisitorReturn(ret);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult DeclareNamedValue(llvm::StringRef name, mlir::Value value) {
    if (symbolTable.count(name))
      return mlir::failure(); // name shadowing not allowed for now
    symbolTable.insert(name, value);
    return mlir::success();
  }

  mlir::Location Loc(Token token) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(token->location.FileName()), token->location.Line(),
                                     token->location.Column());
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::ShapedType TensorType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
  llvm::StringMap<mlir::lox::FuncOp> functionMap;
};

} // namespace

namespace lox::mlir_jit {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> ConvertASTToMLIR(mlir::MLIRContext &context, lox::Module *lox_module) {
  return ASTToMLIR(context).Convert(lox_module->Statements());
}

} // namespace lox::mlir_jit
