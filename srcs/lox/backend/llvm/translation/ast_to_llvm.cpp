#include "ast_to_llvm.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>

#include <memory>

#include "lox/ast/ast.h"
#include "lox/common/finally.h"

class LLVMTranslationError : public lox::LoxError {
 public:
  using LoxError::LoxError;
};

namespace lox::llvm_jit {

/**
 * Create frequently used LLVM types and values.
 */
struct ConstantHelper {
  ConstantHelper(llvm::LLVMContext &context, llvm::Module *module) {
    num_ty = llvm::Type::getDoubleTy(context);
    str_ty = llvm::Type::getInt8PtrTy(context);
    bool_ty = llvm::Type::getInt1Ty(context);
    nil_ty = llvm::Type::getVoidTy(context);

    auto print_num_fn_ty = llvm::FunctionType::get(nil_ty, {num_ty}, false);
    auto print_str_fn_ty = llvm::FunctionType::get(nil_ty, {str_ty}, false);
    auto print_bool_fn_ty = llvm::FunctionType::get(nil_ty, {bool_ty}, false);
    auto print_nil_fn_ty = llvm::FunctionType::get(nil_ty, {}, false);
    module->getOrInsertFunction("__lox_jit_println_num", print_num_fn_ty);
    print_num_fn = module->getFunction("__lox_jit_println_num");
    module->getOrInsertFunction("__lox_jit_println_str", print_str_fn_ty);
    print_str_fn = module->getFunction("__lox_jit_println_str");
    module->getOrInsertFunction("__lox_jit_println_bool", print_bool_fn_ty);
    print_bool_fn = module->getFunction("__lox_jit_println_bool");
    module->getOrInsertFunction("__lox_jit_println_nil", print_nil_fn_ty);
    print_nil_fn = module->getFunction("__lox_jit_println_nil");

    nil = llvm::UndefValue::get(nil_ty);
    nil_num = llvm::UndefValue::get(num_ty);
    nil_str = llvm::UndefValue::get(str_ty);
    nil_bool = llvm::UndefValue::get(bool_ty);
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

  llvm::Function *print_num_fn;
  llvm::Function *print_str_fn;
  llvm::Function *print_bool_fn;
  llvm::Function *print_nil_fn;

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
  explicit ASTToLLVM(llvm::LLVMContext &context) : context_(context), builder_(context) {}

  std::unique_ptr<llvm::Module> Convert(std::vector<StmtPtr> &global_stmts, const std::string &output_module_name) {
    ll_module_ = std::make_unique<llvm::Module>(output_module_name, context_);
    cst_ = std::make_unique<ConstantHelper>(context_, ll_module_.get());
    for (auto &stmt : global_stmts) {
      NoValueVisit(stmt);
    }
    return std::move(ll_module_);
  }

  void Visit(AssignExpr *node) override {
    llvm::Type *ty;
    auto addr = SymAddrLookup(node->attr->name->lexeme, &ty);
    auto new_value = ValueVisit(node->value);
    assert(ty == new_value->getType());  // todo: add implicit type cast support
    builder_.CreateStore(new_value, addr);
    VisitorReturn(builder_.CreateLoad(ty, addr, node->attr->name->lexeme));
  }

  void Visit(LogicalExpr *node) override {
    // short circuit evaluation, we will create two basic blocks, one for else, and one for end
    auto else_bb = llvm::BasicBlock::Create(context_, node->attr->op->lexeme + ".else", current_function_);
    auto end_bb = llvm::BasicBlock::Create(context_, node->attr->op->lexeme + ".end", current_function_);
    auto left_v = ValueVisit(node->left);
    assert(left_v->getType() == cst_->bool_ty);
    auto ret_addr = EntryBlockAlloca(node->attr->op->lexeme, left_v->getType(), left_v);
    if (node->attr->op->type == TokenType::AND) {
      builder_.CreateCondBr(left_v, else_bb, end_bb);
    } else {
      builder_.CreateCondBr(left_v, end_bb, else_bb);
    }
    SwitchBB(else_bb);
    auto else_v = ValueVisit(node->right);
    assert(else_v->getType() == cst_->bool_ty);
    builder_.CreateStore(else_v, ret_addr);
    builder_.CreateBr(end_bb);
    SwitchBB(end_bb);
    VisitorReturn(builder_.CreateLoad(cst_->bool_ty, ret_addr));
  }

  void Visit(BinaryExpr *node) override {
    // todo: support string and boolean type
    auto v0 = ValueVisit(node->left);
    auto v1 = ValueVisit(node->right);
    assert(v0->getType() == v1->getType());
    assert(v0->getType() == cst_->num_ty);
    auto op_code = llvm::Instruction::FAdd;
    auto cmp_code = llvm::CmpInst::FCMP_UEQ;
    const char *code_name = "add";
    bool is_logical = false;
    switch (node->attr->op->type) {
      case TokenType::PLUS: {
        op_code = llvm::Instruction::FAdd;
        code_name = "add";
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
        cmp_code = llvm::CmpInst::FCMP_UEQ;
        code_name = "eq";
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
      ret = llvm::BinaryOperator::Create(op_code, v0, v1, code_name, GetCurrentBB());  // fp
    }
    VisitorReturn(ret);
  }

  void Visit(GroupingExpr *node) override { VisitorReturn(ValueVisit(node->expression)); }

  void Visit(LiteralExpr *node) override {
    switch (node->attr->value->type) {
      case TokenType::NUMBER:
        VisitorReturn(llvm::ConstantFP::get(context_, llvm::APFloat(std::stod(node->attr->value->lexeme))));
      case TokenType::STRING: {
        std::string str_v(node->attr->value->lexeme.begin() + 1, node->attr->value->lexeme.end() - 1);
        VisitorReturn(builder_.CreateGlobalStringPtr(str_v, "", 0, ll_module_.get()));
      }
      case TokenType::NIL:
        VisitorReturn(cst_->nil);
      case TokenType::TRUE_TOKEN:
        VisitorReturn(cst_->true_v);
      case TokenType::FALSE_TOKEN:
        VisitorReturn(cst_->false_v);
      default:
        throw LLVMTranslationError("Not a valid Literal.");
    }
  }

  void Visit(UnaryExpr *node) override {
    auto v = ValueVisit(node->right);
    switch (node->attr->op->type) {
      case TokenType::MINUS: {
        assert(v->getType() == cst_->num_ty);
        VisitorReturn(builder_.CreateNeg(v, "neg"));
      }
      case TokenType::BANG: {
        assert(v->getType() == cst_->bool_ty);
        VisitorReturn(builder_.CreateNot(v, "not"));
      }
      default:
        throw LLVMTranslationError("Not a valid Unary.");
    }
  }

  void Visit(CallExpr *node) override {
    auto fn_name = node->callee->DynAs<VariableExpr>()->attr->name->lexeme;
    auto callee_fn = ll_module_->getFunction(fn_name);
    if (!callee_fn) throw LLVMTranslationError("Unknown function referenced");

    auto &args = node->comma_expr_args->As<CommaExpr>()->elements;
    if (callee_fn->arg_size() != args.size()) throw LLVMTranslationError(std::string("Incorrect arguments number"));

    std::vector<llvm::Value *> args_value;
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      args_value.push_back(ValueVisit(args[i]));
      if (!args_value.back()) throw LLVMTranslationError("Incorrect arguments value");
    }

    VisitorReturn(builder_.CreateCall(callee_fn, args_value, std::string("call_") + fn_name));
  }

  void Visit(GetAttrExpr *node) override { throw LLVMTranslationError("GetAttrExpr is not supported yet"); }

  void Visit(SetAttrExpr *node) override { throw LLVMTranslationError("SetAttrExpr is not supported yet"); }

  void Visit(VariableExpr *node) override {
    llvm::Type *ty = nullptr;
    auto addr = SymAddrLookup(node->attr->name->lexeme, &ty);
    VisitorReturn(builder_.CreateLoad(ty, addr, node->attr->name->lexeme));
  }

  void Visit(CommaExpr *node) override { throw LLVMTranslationError("CommaExpr is not supported yet"); }

  void Visit(ListExpr *node) override { throw LLVMTranslationError("ListExpr is not supported yet"); }

  void Visit(GetItemExpr *node) override { throw LLVMTranslationError("GetItemExpr is not supported yet"); }

  void Visit(TensorExpr *node) override { throw LLVMTranslationError("TensorExpr is not supported yet"); }

  void Visit(VarDeclStmt *node) override {
    // Create global variable when at the global scope, or create local variable
    llvm::Value *init_v = nullptr;
    if (node->initializer) {
      init_v = ValueVisit(node->initializer);
    }
    if (IsAtGlobal()) {
      auto *var_type = GetType(node->attr->type_hint);
      assert(ll_module_->getGlobalVariable(node->attr->name->lexeme) == nullptr);
      ll_module_->getOrInsertGlobal(node->attr->name->lexeme, var_type);
      auto global_var = ll_module_->getNamedGlobal(node->attr->name->lexeme);
      if (init_v) {
        global_var->setInitializer(llvm::dyn_cast<llvm::Constant>(init_v));
      }
    } else {
      NamedEntryBlockAlloca(node->attr->name->lexeme, GetType(node->attr->type_hint), init_v);
    }
  }

  void Visit(WhileStmt *node) override {
    llvm::BasicBlock *while_cond_bb = llvm::BasicBlock::Create(context_, "while.cond", current_function_);
    llvm::BasicBlock *while_body_bb = llvm::BasicBlock::Create(context_, "while.body", current_function_);
    llvm::BasicBlock *while_end_bb = llvm::BasicBlock::Create(context_, "while.end", current_function_);
    // branch to while_cond_bb
    ScopeGuard guard(local_sym_table_);  // while in a new scope
    builder_.CreateBr(while_cond_bb);

    SwitchBB(while_cond_bb);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, while_body_bb, while_end_bb);

    break_targets_.push_back(while_end_bb);
    continue_targets_.push_back(while_cond_bb);
    SwitchBB(while_body_bb);
    NoValueVisit(node->body);
    builder_.CreateBr(while_cond_bb);
    break_targets_.pop_back();
    continue_targets_.pop_back();

    SwitchBB(while_end_bb);
  }

  void Visit(ForStmt *node) override {
    llvm::BasicBlock *for_init_bb = llvm::BasicBlock::Create(context_, "for.init", current_function_);
    llvm::BasicBlock *for_cond_bb = llvm::BasicBlock::Create(context_, "for.cond", current_function_);
    llvm::BasicBlock *for_inc_bb = llvm::BasicBlock::Create(context_, "for.inc", current_function_);
    llvm::BasicBlock *for_body_bb = llvm::BasicBlock::Create(context_, "for.body", current_function_);
    llvm::BasicBlock *for_end_bb = llvm::BasicBlock::Create(context_, "for.end", current_function_);
    // branch to for_cond_bb
    ScopeGuard guard(local_sym_table_);  // for in a new scope
    builder_.CreateBr(for_init_bb);

    SwitchBB(for_init_bb);
    NoValueVisit(node->initializer);
    builder_.CreateBr(for_cond_bb);

    SwitchBB(for_cond_bb);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, for_body_bb, for_end_bb);

    break_targets_.push_back(for_end_bb);
    continue_targets_.push_back(for_inc_bb);
    SwitchBB(for_body_bb);
    NoValueVisit(node->body);
    builder_.CreateBr(for_inc_bb);
    break_targets_.pop_back();
    continue_targets_.pop_back();

    SwitchBB(for_inc_bb);
    ValueVisit(node->increment);  // discard value
    builder_.CreateBr(for_cond_bb);

    SwitchBB(for_end_bb);
  }

  void Visit(ExprStmt *node) override {
    ValueVisit(node->expression);  // discard value
  }

  void Visit(FunctionStmt *node) override {
    assert(IsAtGlobal());  // only global functions are allowed.

    llvm::Type *ret_ty = cst_->nil_ty;
    if (node->attr->ret_type_hint) {
      ret_ty = GetType(node->attr->ret_type_hint);
    }
    std::vector<llvm::Type *> arg_tys;
    std::vector<std::string> arg_names;
    if (node->comma_expr_params) {
      for (auto &arg : node->comma_expr_params->As<CommaExpr>()->elements) {
        arg_tys.push_back(GetType(arg->As<VariableExpr>()->attr->type_hint));
        arg_names.push_back(arg->As<VariableExpr>()->attr->name->lexeme);
      }
    }
    llvm::FunctionType *fn_type = llvm::FunctionType::get(ret_ty, arg_tys, false);
    llvm::Function *fn =
        llvm::Function::Create(fn_type, llvm::GlobalValue::ExternalLinkage, node->attr->name->lexeme, ll_module_.get());

    // switch current_function
    current_function_ = fn;
    lox::Finally finally([this]() { current_function_ = nullptr; });

    ScopeGuard new_scope(local_sym_table_);
    // add entry bb and switch insert point to it
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(context_, "entry", fn);
    SwitchBB(BB);

    // bind value to arg names
    for (auto &arg : fn->args()) {
      arg.setName(arg_names[arg.getArgNo()]);
      NamedEntryBlockAlloca(arg_names[arg.getArgNo()], arg.getType(), &arg);
    }

    for (auto &stmt : node->body) {
      NoValueVisit(stmt);
    }
    // always inject a return
    if (ret_ty != cst_->nil_ty) {
      builder_.CreateRet(llvm::Constant::getNullValue(ret_ty));
    } else {
      builder_.CreateRetVoid();
    }

    llvm::verifyFunction(*fn);
  }

  void Visit(ClassStmt *node) override { throw LLVMTranslationError("class is not supported"); }

  void Visit(PrintStmt *node) override {
    llvm::Value *v = ValueVisit(node->expression);
    if (v->getType() == cst_->num_ty) {
      builder_.CreateCall(cst_->print_num_fn, v);
    } else if (v->getType() == cst_->str_ty) {
      builder_.CreateCall(cst_->print_str_fn, v);
    } else if (v->getType() == cst_->bool_ty) {
      builder_.CreateCall(cst_->print_bool_fn, v);
    } else if (v->getType() == cst_->nil_ty) {
      builder_.CreateCall(cst_->print_nil_fn);
    } else {
      throw LLVMTranslationError("unsupported type for print");
    }
  }

  void Visit(ReturnStmt *node) override {
    if (node->value) {
      auto value = ValueVisit(node->value);
      builder_.CreateRet(value);
    } else {
      builder_.CreateRetVoid();
    }
  }

  void Visit(BlockStmt *node) override {
    ScopeGuard guard(local_sym_table_);
    for (auto &stmt : node->statements) {
      NoValueVisit(stmt);
    }
  }

  void Visit(IfStmt *node) override {
    llvm::BasicBlock *if_then_bb = llvm::BasicBlock::Create(context_, "if.then", current_function_);
    llvm::BasicBlock *if_else_bb = llvm::BasicBlock::Create(context_, "if.else", current_function_);
    llvm::BasicBlock *if_end_bb = llvm::BasicBlock::Create(context_, "if.end", current_function_);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, if_then_bb, if_else_bb);

    SwitchBB(if_then_bb);
    NoValueVisit(node->then_branch);
    builder_.CreateBr(if_end_bb);

    SwitchBB(if_else_bb);
    if (node->else_branch) {
      NoValueVisit(node->else_branch);
    }
    builder_.CreateBr(if_end_bb);

    SwitchBB(if_end_bb);
  }

  void Visit(BreakStmt *node) override {
    auto break_type = node->attr->src_token->type;
    if (break_type == TokenType::BREAK) {
      assert(!break_targets_.empty());
      builder_.CreateBr(break_targets_.back());
    } else if (break_type == TokenType::CONTINUE) {
      assert(!continue_targets_.empty());
      builder_.CreateBr(continue_targets_.back());
    } else {
      throw LLVMTranslationError("unsupported break type");
    }
  }

 protected:
  void SwitchBB(llvm::BasicBlock *bb) { builder_.SetInsertPoint(bb); }
  llvm::BasicBlock *GetCurrentBB() { return builder_.GetInsertBlock(); }

  bool IsAtGlobal() { return current_function_ == nullptr; }

  llvm::LLVMContext &context_;
  std::unique_ptr<llvm::Module> ll_module_;
  std::unique_ptr<ConstantHelper> cst_;
  llvm::IRBuilder<> builder_;
  llvm::Function *current_function_ = nullptr;
  std::vector<llvm::BasicBlock *> break_targets_;
  std::vector<llvm::BasicBlock *> continue_targets_;

  llvm::ScopedHashTable<llvm::StringRef, llvm::AllocaInst *>
      local_sym_table_;  // map from name to latest alloca address
  struct ScopeGuard {
    ScopeGuard(llvm::ScopedHashTable<llvm::StringRef, llvm::AllocaInst *> &symtable) : scope(symtable) {}
    llvm::ScopedHashTableScope<llvm::StringRef, llvm::AllocaInst *> scope;
  };

  llvm::AllocaInst *EntryBlockAlloca(const std::string &var_name, llvm::Type *ty, llvm::Value *init_value) {
    assert(!IsAtGlobal());
    // use tmp builder to avoid basic block switch
    llvm::IRBuilder<> tmp_builder(&current_function_->getEntryBlock(), current_function_->getEntryBlock().begin());
    auto ret_inst = tmp_builder.CreateAlloca(ty, 0, var_name.c_str());
    if (init_value) {
      builder_.CreateStore(init_value, ret_inst);
    }
    return ret_inst;
  }
  llvm::AllocaInst *NamedEntryBlockAlloca(const std::string &name, llvm::Type *ty, llvm::Value *init_value) {
    auto addr = EntryBlockAlloca(name, ty, init_value);
    local_sym_table_.insert(name, addr);
    return addr;
  }
  llvm::Value *SymAddrLookup(const std::string &name, llvm::Type **o_ty = nullptr) {
    if (!IsAtGlobal() && local_sym_table_.count(name)) {
      auto ret = local_sym_table_.lookup(name);
      if (o_ty) {
        *o_ty = ret->getAllocatedType();
      }
      return ret;
    } else {
      auto ret = ll_module_->getGlobalVariable(name);
      assert(ret);
      if (o_ty) {
        *o_ty = ret->getValueType();
      }
      return ret;
    }
  }
  llvm::Type *GetType(const std::string &name) const {
    if (name == "float") {
      return cst_->num_ty;
    } else if (name == "bool") {
      return cst_->bool_ty;
    } else if (name == "str") {
      return cst_->str_ty;
    } else {
      throw LLVMTranslationError("unknown type: " + name);
    }
  }
  llvm::Type *GetType(const lox::Token &token) const {
    if (!(bool)token) {
      throw LLVMTranslationError("No type hint found");
    }
    return GetType(token->lexeme);
  }
};

std::unique_ptr<llvm::Module> ConvertASTToLLVM(llvm::LLVMContext &context, lox::Module *lox_module) {
  return ASTToLLVM(context).Convert(lox_module->Statements(), "main_module");
}

}  // namespace lox::llvm_jit