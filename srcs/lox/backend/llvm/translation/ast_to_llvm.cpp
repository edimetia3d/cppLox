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
  ConstantHelper(llvm::LLVMContext &context) {
    num_ty = llvm::Type::getDoubleTy(context);
    str_ty = llvm::Type::getInt8PtrTy(context);
    bool_ty = llvm::Type::getInt1Ty(context);
    nil_ty = llvm::Type::getVoidTy(context);
    i8_ty = llvm::Type::getInt8Ty(context);

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
  llvm::Type *i8_ty;

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

  bool ShouldBeInDefModule(StmtPtr &stmt) const { return stmt->DynAs<VarDeclStmt>() || stmt->DynAs<FunctionStmt>(); }

  std::unique_ptr<llvm::Module> MakeDefModule(std::vector<StmtPtr> &global_stmts,
                                              const std::string &output_module_name) {
    auto def_module = std::make_unique<llvm::Module>(output_module_name, context_);
    active_module_ = def_module.get();
    lox::Finally finally([this]() { active_module_ = nullptr; });
    for (auto &stmt : global_stmts) {
      if (ShouldBeInDefModule(stmt)) {
        NoValueVisit(stmt);
      }
    }
    return def_module;
  }

  std::unique_ptr<llvm::Module> MakeInitModule(std::vector<StmtPtr> &global_stmts,
                                               const std::string &output_module_name) {
    auto init_module = std::make_unique<llvm::Module>(output_module_name, context_);
    active_module_ = init_module.get();
    lox::Finally finally([this]() { active_module_ = nullptr; });

    llvm::FunctionType *fn_type = llvm::FunctionType::get(cst_->nil_ty, {}, false);
    active_module_->getOrInsertFunction("__lox_init_module", fn_type);
    // NOTE : DO NOT add __lox_init_module into known global , for every module will have their own __lox_init_module
    llvm::Function *fn = active_module_->getFunction("__lox_init_module");
    fn->setLinkage(llvm::GlobalValue::ExternalLinkage);

    // switch current_function
    current_function_ = fn;
    lox::Finally fn_finally([this]() { current_function_ = nullptr; });

    ScopeGuard new_scope(local_sym_table_);
    // add entry bb and switch insert point to it
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(context_, "entry", fn);
    SwitchBB(BB);

    for (auto &stmt : global_stmts) {
      if (!ShouldBeInDefModule(stmt)) {
        NoValueVisit(stmt);
      }
    }

    AddReturn(nullptr);
    FunctionBodyDone(*fn);

    return init_module;
  }

  ConvertedModule Convert(std::vector<StmtPtr> &global_stmts, const std::string &output_module_name,
                          KnownGlobalSymbol *known_global_symbol) {
    known_global_symbol_ = known_global_symbol;
    cst_ = std::make_unique<ConstantHelper>(context_);

    auto def_module = MakeDefModule(global_stmts, output_module_name);
    auto init_module = MakeInitModule(global_stmts, output_module_name + "_init");
    return {std::move(def_module), std::move(init_module)};
  }

  void Visit(AssignExpr *node) override {
    llvm::Type *ty;
    auto addr = SymAddrLookup(node->attr->name->lexeme, &ty);
    auto new_value = ValueVisit(node->value);
    assert(ty == new_value->getType()); // todo: add implicit type cast support
    builder_.CreateStore(new_value, addr);
    VisitorReturn(builder_.CreateLoad(ty, addr, node->attr->name->lexeme.Str()));
  }

  void Visit(LogicalExpr *node) override {
    // short circuit evaluation, we will create two basic blocks, one for else, and one for end
    auto else_bb = llvm::BasicBlock::Create(context_, node->attr->op->lexeme.Str() + ".else", current_function_);
    auto end_bb = llvm::BasicBlock::Create(context_, node->attr->op->lexeme.Str() + ".end", current_function_);
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
    llvm::Value *ret = NumBinary(node->attr->op->type, v0, v1);
    VisitorReturn(ret);
  }

  llvm::Value *NumBinary(TokenType op, llvm::Value *v0, llvm::Value *v1) {
    auto op_code = llvm::Instruction::FAdd;
    auto cmp_code = llvm::CmpInst::FCMP_UEQ;
    const char *code_name = "add";
    bool is_logical = false;
#define MATH_OP(token_name, llvm_op_code, code_name_str)                                                               \
  case TokenType::token_name: {                                                                                        \
    op_code = llvm::Instruction::llvm_op_code;                                                                         \
    code_name = code_name_str;                                                                                         \
    break;                                                                                                             \
  }

#define LOGICAL_OP(token_name, llvm_op_code, code_name_str)                                                            \
  case TokenType::token_name: {                                                                                        \
    cmp_code = llvm::CmpInst::llvm_op_code;                                                                            \
    code_name = code_name_str;                                                                                         \
    is_logical = true;                                                                                                 \
    break;                                                                                                             \
  }

    switch (op) {
      MATH_OP(PLUS, FAdd, "add")
      MATH_OP(MINUS, FSub, "sub")
      MATH_OP(STAR, FMul, "mul")
      MATH_OP(SLASH, FDiv, "div")
      LOGICAL_OP(EQUAL_EQUAL, FCMP_UEQ, "eq")
      LOGICAL_OP(BANG_EQUAL, FCMP_UNE, "ne")
      LOGICAL_OP(LESS, FCMP_ULT, "lt")
      LOGICAL_OP(LESS_EQUAL, FCMP_ULE, "le")
      LOGICAL_OP(GREATER, FCMP_UGT, "gt")
      LOGICAL_OP(GREATER_EQUAL, FCMP_UGE, "ge")

    default:
      throw LLVMTranslationError("not supported op");
    }
#undef MATH_OP
#undef LOGICAL_OP
    llvm::Value *ret = nullptr;
    if (is_logical) {
      ret = builder_.CreateFCmp(cmp_code, v0, v1, code_name); // i1
    } else {
      ret = llvm::BinaryOperator::Create(op_code, v0, v1, code_name, GetCurrentBB()); // fp
    }
    return ret;
  }

  void Visit(GroupingExpr *node) override { VisitorReturn(ValueVisit(node->expression)); }

  void Visit(LiteralExpr *node) override {
    switch (node->attr->value->type) {
    case TokenType::NUMBER:
      VisitorReturn(llvm::ConstantFP::get(context_, llvm::APFloat(std::stod(node->attr->value->lexeme))));
    case TokenType::STRING: {
      std::string str_v(node->attr->value->lexeme.Data() + 1, node->attr->value->lexeme.End() - 1);
      VisitorReturn(builder_.CreateGlobalStringPtr(str_v, "", 0, active_module_));
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
    llvm::Function *callee_fn = FunctionLookUp(fn_name);

    auto &args = node->comma_expr_args->As<CommaExpr>()->elements;
    if (callee_fn->arg_size() != args.size())
      throw LLVMTranslationError(std::string("Incorrect arguments number"));

    std::vector<llvm::Value *> args_value;
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      args_value.push_back(ValueVisit(args[i]));
      if (!args_value.back())
        throw LLVMTranslationError("Incorrect arguments value");
    }
    std::string ret_name;
    if (!callee_fn->getReturnType()->isVoidTy()) {
      ret_name = std::string("call_") + fn_name.Str();
    }
    VisitorReturn(builder_.CreateCall(callee_fn, args_value, ret_name));
  }

  void Visit(GetAttrExpr *node) override { throw LLVMTranslationError("GetAttrExpr is not supported yet"); }

  void Visit(SetAttrExpr *node) override { throw LLVMTranslationError("SetAttrExpr is not supported yet"); }

  void Visit(VariableExpr *node) override {
    llvm::Type *ty = nullptr;
    auto addr = SymAddrLookup(node->attr->name->lexeme, &ty);
    VisitorReturn(builder_.CreateLoad(ty, addr, node->attr->name->lexeme.Str()));
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
      assert(active_module_->getGlobalVariable(node->attr->name->lexeme.Str()) == nullptr);
      active_module_->getOrInsertGlobal(node->attr->name->lexeme.Str(), var_type);
      auto global_var = active_module_->getNamedGlobal(node->attr->name->lexeme.Str());
      if (init_v) {
        global_var->setInitializer(llvm::dyn_cast<llvm::Constant>(init_v));
      }
      known_global_symbol_->insert({node->attr->name->lexeme, var_type});
    } else {
      NamedEntryBlockAlloca(node->attr->name->lexeme, GetType(node->attr->type_hint), init_v);
    }
  }

  void Visit(WhileStmt *node) override {
    llvm::BasicBlock *while_cond_bb = llvm::BasicBlock::Create(context_, "while.cond", current_function_);
    llvm::BasicBlock *while_body_bb = llvm::BasicBlock::Create(context_, "while.body", current_function_);
    llvm::BasicBlock *while_end_bb = llvm::BasicBlock::Create(context_, "while.end", current_function_);
    // branch to while_cond_bb
    ScopeGuard guard(local_sym_table_); // while in a new scope
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
    ScopeGuard guard(local_sym_table_); // for in a new scope
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
    ValueVisit(node->increment); // discard value
    builder_.CreateBr(for_cond_bb);

    SwitchBB(for_end_bb);
  }

  void Visit(ExprStmt *node) override {
    ValueVisit(node->expression); // discard value
  }

  void Visit(FunctionStmt *node) override {
    assert(IsAtGlobal()); // only global functions are allowed.

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
    active_module_->getOrInsertFunction(node->attr->name->lexeme.Str(), fn_type);
    known_global_symbol_->insert({node->attr->name->lexeme, fn_type});
    if (node->attr->is_decl) {
      return;
    }
    llvm::Function *fn = active_module_->getFunction(node->attr->name->lexeme.Str());
    fn->setLinkage(llvm::GlobalValue::ExternalLinkage);

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

    if (ret_ty != cst_->nil_ty) {
      AddReturn(llvm::Constant::getNullValue(ret_ty));
    } else {
      AddReturn(nullptr);
    }
    FunctionBodyDone(*fn);
  }

  void Visit(ClassStmt *node) override { throw LLVMTranslationError("class is not supported"); }

  void Visit(PrintStmt *node) override {
    llvm::Value *v = ValueVisit(node->expression);
    if (v->getType() == cst_->num_ty) {
      builder_.CreateCall(FunctionLookUp("__lox_jit_println_num"), v);
    } else if (v->getType() == cst_->str_ty) {
      builder_.CreateCall(FunctionLookUp("__lox_jit_println_str"), v);
    } else if (v->getType() == cst_->bool_ty) {
      auto cast = builder_.CreateIntCast(v, cst_->i8_ty, false, v->getName().str() + ".cast");
      builder_.CreateCall(FunctionLookUp("__lox_jit_println_bool"), cast);
    } else if (v->getType() == cst_->nil_ty) {
      builder_.CreateCall(FunctionLookUp("__lox_jit_println_nil"));
    } else {
      throw LLVMTranslationError("unsupported type for print");
    }
  }

  void Visit(ReturnStmt *node) override {
    llvm::Value *value = nullptr;
    if (node->value) {
      value = ValueVisit(node->value);
    }
    AddReturn(value);
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
  llvm::Module *active_module_;
  KnownGlobalSymbol *known_global_symbol_;
  std::unique_ptr<ConstantHelper> cst_;
  llvm::IRBuilder<> builder_;
  llvm::Function *current_function_ = nullptr;
  std::vector<llvm::BasicBlock *> possible_exit_bb_; // BBs that we are not sure if they will terminate
  std::vector<llvm::BasicBlock *> break_targets_;
  std::vector<llvm::BasicBlock *> continue_targets_;

  llvm::ScopedHashTable<llvm::StringRef, llvm::AllocaInst *> local_sym_table_; // map from name to latest alloca address
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
      auto ret = active_module_->getGlobalVariable(name);
      if (!ret) {
        if (known_global_symbol_->contains(name)) {
          active_module_->getOrInsertGlobal(name, known_global_symbol_->at(name));
          ret = active_module_->getGlobalVariable(name);
        } else {
          throw LLVMTranslationError("unknown symbol: " + name);
        }
      }
      if (o_ty) {
        *o_ty = ret->getValueType();
      }
      return ret;
    }
  }

  llvm::Function *FunctionLookUp(const std::string &fn_name) {
    auto callee_fn = active_module_->getFunction(fn_name);
    if (!callee_fn) {
      if (known_global_symbol_->contains(fn_name)) {
        active_module_->getOrInsertFunction(fn_name, llvm::cast<llvm::FunctionType>(known_global_symbol_->at(fn_name)));
        callee_fn = active_module_->getFunction(fn_name);
      } else {
        throw LLVMTranslationError("Unknown function referenced");
      }
    }
    return callee_fn;
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

  void AddReturn(llvm::Value *value) {
    assert(!IsAtGlobal());
    if (value && !value->getType()->isVoidTy()) {
      builder_.CreateRet(value);
    } else {
      builder_.CreateRetVoid();
    }
    auto tmp_bb = llvm::BasicBlock::Create(context_, "tmp_exit", current_function_);
    // There might still be stmt after return expression, however, the bb that `return` belongs to is already terminated
    // so we need to switch to a new bb. However, we don't know whether this new bb will be terminated or not
    // so we need to keep track of it by adding it to possible_exit_bb_
    SwitchBB(tmp_bb);
    possible_exit_bb_.push_back(tmp_bb);
  }

  void TerminateAllExitBlocks() {
    auto bak = builder_.GetInsertBlock();
    for (auto &bb : possible_exit_bb_) {
      SwitchBB(bb);
      if (bb->getTerminator() == nullptr) {
        builder_.CreateBr(bb);
      }
    }
    possible_exit_bb_.clear();
    SwitchBB(bak);
  }

  void FunctionBodyDone(const llvm::Function &F) {
    // When a function is done, we should make sure that all exit blocks are terminated
    TerminateAllExitBlocks();
    llvm::verifyFunction(F);
  }
};

ConvertedModule ConvertASTToLLVM(llvm::LLVMContext &context, lox::Module *lox_module,
                                 KnownGlobalSymbol *known_global_symbol) {
  return ASTToLLVM(context).Convert(lox_module->Statements(), "main_module", known_global_symbol);
}

} // namespace lox::llvm_jit