#include "ast_to_llvm.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>

#include <memory>

#include "lox/ast/ast.h"

class LLVMTranslationError : public lox::LoxError {
 public:
  using LoxError::LoxError;
};

namespace lox::llvm_jit {

/**
 * Create frequently used LLVM types and values.
 */
struct ConstantHelper {
  ConstantHelper(llvm::LLVMContext &context) {
    num_ty = llvm::Type::getDoubleTy(context);
    str_ty = llvm::Type::getInt8PtrTy(context);
    bool_ty = llvm::Type::getInt1Ty(context);
    nil_ty = llvm::Type::getVoidTy(context);

    nil = llvm::Constant::getNullValue(nil_ty);
    nil_num = llvm::Constant::getNullValue(num_ty);
    nil_str = llvm::Constant::getNullValue(str_ty);
    nil_bool = llvm::Constant::getNullValue(bool_ty);
    true_v = llvm::ConstantInt::get(bool_ty, 1);
    false_v = llvm::ConstantInt::get(bool_ty, 0);
    num_0 = llvm::ConstantFP::get(num_ty, 0.0);
    num_1 = llvm::ConstantFP::get(num_ty, 1.0);
    num_2 = llvm::ConstantFP::get(num_ty, 2.0);
  }
  llvm::Type *num_ty;
  llvm::Type *str_ty;
  llvm::Type *bool_ty;
  llvm::Type *nil_ty;

  llvm::Value *nil;
  llvm::Value *nil_num;
  llvm::Value *nil_str;
  llvm::Value *nil_bool;
  llvm::Value *true_v;
  llvm::Value *false_v;
  llvm::Value *num_0;
  llvm::Value *num_1;
  llvm::Value *num_2;
};

class ASTToLLVM : public lox::ASTNodeVisitor<llvm::Value *> {
 public:
  explicit ASTToLLVM(llvm::LLVMContext &context) : cst_(context), context_(context), builder_(context) {}

  std::unique_ptr<llvm::Module> Convert(std::vector<StmtPtr> &global_stmts, const std::string &output_module_name) {
    ll_module_ = std::make_unique<llvm::Module>(output_module_name, context_);
    for (auto &stmt : global_stmts) {
      NoValueVisit(stmt);
    }
    return std::move(ll_module_);
  }

  void Visit(AssignExpr *node) override { auto value = ValueVisit(node->value); }

  void Visit(LogicalExpr *node) override {}

  void Visit(BinaryExpr *node) override {
    // only numberic type can be used in binary expression for now
    // todo: support string and boolean type
    auto v0 = ValueVisit(node->left);
    auto v1 = ValueVisit(node->right);
    assert(v0->getType() == v1->getType());
    assert(v0->getType() == cst_.num_ty);
    auto op_code = llvm::Instruction::FAdd;
    auto cmp_code = llvm::CmpInst::FCMP_UEQ;
    const char *code_name = "add";
    bool is_logical = false;
    switch (node->attr->op->type) {
      case TokenType::PLUS: {
        break;
      }
      case TokenType::MINUS: {
        op_code = llvm::Instruction::FSub;
        code_name = "sub";
        break;
      }
      case TokenType::STAR: {
        op_code = llvm::Instruction::FMul;
        code_name = "mul";
        break;
      }
      case TokenType::SLASH: {
        op_code = llvm::Instruction::FDiv;
        code_name = "div";
        break;
      }
      case TokenType::EQUAL_EQUAL: {
        is_logical = true;
        break;
      }
      case TokenType::BANG_EQUAL: {
        cmp_code = llvm::CmpInst::FCMP_UNE;
        code_name = "ne";
        is_logical = true;
        break;
      }
      case TokenType::LESS: {
        cmp_code = llvm::CmpInst::FCMP_ULT;
        code_name = "lt";
        is_logical = true;
        break;
      }
      case TokenType::LESS_EQUAL: {
        cmp_code = llvm::CmpInst::FCMP_ULE;
        code_name = "le";
        is_logical = true;
        break;
      }
      case TokenType::GREATER: {
        cmp_code = llvm::CmpInst::FCMP_UGT;
        code_name = "gt";
        is_logical = true;
        break;
      }
      case TokenType::GREATER_EQUAL: {
        cmp_code = llvm::CmpInst::FCMP_UGE;
        code_name = "ge";
        is_logical = true;
        break;
      }

      default:
        throw LLVMTranslationError("not supported op");
    }
    llvm::Value *ret = nullptr;
    if (is_logical) {
      ret = builder_.CreateFCmp(cmp_code, v0, v1, code_name);  // i1
    } else {
      ret = llvm::BinaryOperator::Create(op_code, v0, v1, code_name);  // fp
    }
    VisitorReturn(ret);
  }

  void Visit(GroupingExpr *node) override { VisitorReturn(ValueVisit(node->expression)); }

  void Visit(LiteralExpr *node) override {
    switch (node->attr->value->type) {
      case TokenType::NUMBER:
        VisitorReturn(llvm::ConstantFP::get(context_, llvm::APFloat(std::stod(node->attr->value->lexeme))));
      case TokenType::STRING:
        VisitorReturn(llvm::ConstantDataArray::getString(context_, node->attr->value->lexeme, true));
      case TokenType::NIL:
        VisitorReturn(cst_.nil);
      case TokenType::TRUE_TOKEN:
        VisitorReturn(llvm::ConstantInt::getTrue(context_));
      case TokenType::FALSE_TOKEN:
        VisitorReturn(llvm::ConstantInt::getFalse(context_));
      default:
        throw LLVMTranslationError("Not a valid Literal.");
    }
  }

  void Visit(UnaryExpr *node) override {}

  void Visit(CallExpr *node) override {
    // Look up the name in the global module table.
    // fixme: function pointer and lambda function are not supported yet.
    auto *calle_fn = ll_module_->getFunction(node->callee->DynAs<VariableExpr>()->attr->name->lexeme);
    if (!calle_fn) throw LLVMTranslationError("Unknown function referenced");

    // If argument mismatch error.
    auto &args = node->comma_expr_args->As<CommaExpr>()->elements;
    if (calle_fn->arg_size() != args.size()) throw LLVMTranslationError("Incorrect # arguments passed");

    std::vector<llvm::Value *> args_value;
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      args_value.push_back(ValueVisit(args[i]));
      if (!args_value.back()) throw LLVMTranslationError("Incorrect # arguments values");
    }

    VisitorReturn(builder_.CreateCall(calle_fn, args_value, "calltmp"));
  }

  void Visit(GetAttrExpr *node) override {}

  void Visit(SetAttrExpr *node) override {}

  void Visit(VariableExpr *node) override {}

  void Visit(CommaExpr *node) override {}

  void Visit(ListExpr *node) override {}

  void Visit(GetItemExpr *node) override {}

  void Visit(TensorExpr *node) override {}

  void Visit(VarDeclStmt *node) override { auto value = ValueVisit(node->initializer); }

  void Visit(WhileStmt *node) override {
    auto fn = builder_.GetInsertBlock()->getParent();
    llvm::BasicBlock *while_cond_bb = llvm::BasicBlock::Create(context_, "while.cond", fn);
    llvm::BasicBlock *while_body_bb = llvm::BasicBlock::Create(context_, "while.body", fn);
    llvm::BasicBlock *after_while_bb = llvm::BasicBlock::Create(context_, "after.while", fn);
    // branch to while_cond_bb
    builder_.CreateBr(while_cond_bb);

    // seal the current active bb

    // switch to while_cond_bb
    SwitchBB(while_cond_bb);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, while_body_bb, after_while_bb);

    // switch to while_body_bb
    SwitchBB(while_body_bb);
    NoValueVisit(node->body);
    builder_.CreateBr(while_cond_bb);

    // switch back
    SwitchBB(after_while_bb);
  }

  void Visit(ForStmt *node) override {}

  void Visit(ExprStmt *node) override { NoValueVisit(node->expression); }

  void Visit(FunctionStmt *node) override {
    assert(IsAtGlobal());  // only global functions are allowed.

    assert(node->attr->name->lexeme == "main");  // only a main function with no arguments is supported by now
    assert(!node->comma_expr_params || node->comma_expr_params->DynAs<CommaExpr>()->elements.empty());

    // create fn
    assert(node->attr->name->lexeme == "main");  // only main is supported by now
    llvm::FunctionType *fn_type = llvm::FunctionType::get(cst_.num_ty, {}, false);
    llvm::Function *fn =
        llvm::Function::Create(fn_type, llvm::GlobalValue::ExternalLinkage, node->attr->name->lexeme, ll_module_.get());

    // set arg names
    for (auto &arg : fn->args()) {
      // todo: fill name from node->comma_expr_params
      arg.setName("arg." + std::to_string(arg.getArgNo()));
    }

    // add new function to back as active function
    function_hierarchy_.push_back(fn);

    // create a guard to switch back to the current BB after the function
    llvm::IRBuilder<>::InsertPointGuard guard(builder_);
    // add entry bb and switch insert point to it
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(context_, "entry", fn);

    SwitchBB(BB);

    // bind value to arg names
    for (unsigned i = 0, e = fn->arg_size(); i != e; ++i) {
      // todo : update me
    }

    for (auto &stmt : node->body) {
      NoValueVisit(stmt);
    }
    // todo: check return type and inject a valid return value

    // pop the function to disable it
    function_hierarchy_.pop_back();
    llvm::verifyFunction(*fn);
  }

  void Visit(ClassStmt *node) override {}

  void Visit(PrintStmt *node) override {}

  void Visit(ReturnStmt *node) override {
    if (node->value) {
      auto value = ValueVisit(node->value);
      assert(value->getType() == cst_.num_ty);  // only number supported by now
      builder_.CreateRet(value);
    } else {
      builder_.CreateRetVoid();
    }
  }

  void Visit(BlockStmt *node) override {}

  void Visit(IfStmt *node) override {}

  void Visit(BreakStmt *node) override {}

 protected:
  void SwitchBB(llvm::BasicBlock *bb) {
    active_bb_ = bb;
    builder_.SetInsertPoint(active_bb_);
  }

  bool IsAtGlobal() { return function_hierarchy_.size() == 0; }

  const ConstantHelper cst_;
  llvm::LLVMContext &context_;
  std::unique_ptr<llvm::Module> ll_module_;
  llvm::IRBuilder<> builder_;
  std::vector<llvm::Function *> function_hierarchy_;
  llvm::BasicBlock *active_bb_;
};

std::unique_ptr<llvm::Module> ConvertASTToLLVM(llvm::LLVMContext &context, lox::Module *lox_module) {
  auto ret = ASTToLLVM(context).Convert(lox_module->Statements(), "main_module");
  ret->print(llvm::outs(), nullptr);  // todo: remove debug print later
  return ret;
}

}  // namespace lox::llvm_jit