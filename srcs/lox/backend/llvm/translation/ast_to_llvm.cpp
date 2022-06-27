#include "ast_to_llvm.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

#include <memory>

#include "lox/ast/ast.h"

class LLVMTranslationError : public lox::LoxError {
 public:
  using LoxError::LoxError;
};

namespace lox::llvm_jit {

class BasicBlockSymResolver;

class BasicBlockSymResolverManager {
 public:
  BasicBlockSymResolver &GetResolver(llvm::BasicBlock *);
  llvm::DenseMap<llvm::BasicBlock *, std::shared_ptr<BasicBlockSymResolver>> map;
};

/**
 * A helper class to find the latest value of a symbol in the specific BB.
 * It will search values in BB first, if not found, it will try to find value in predecessors by creating a phi node.
 * Note:
 * 1. There should always be a value to be found, or it is a semantic error.
 * 2. Phi node may created empty, and be updated when Seal() is called.
 *
 */
class BasicBlockSymResolver {
 public:
  BasicBlockSymResolver(BasicBlockSymResolverManager *mgr, llvm::BasicBlock *source_bb)
      : resovle_manager_(mgr), source_bb_(source_bb), sealed(false) {}

  void write_local_variable(llvm::StringRef var_name, llvm::Value *value) { values_[var_name] = value; }
  llvm::Value *readLocalVariable(llvm::StringRef var_name) {
    auto Val = values_.find(var_name);
    if (Val != values_.end()) return Val->second;
    // when code goes here, a variable could only be found in other bb, so a phi will be needed.
    // after the phi node is created, search in BB will never go here again
    return read_var_from_predecessor(var_name);
  }
  llvm::Value *read_var_from_predecessor(llvm::StringRef var_name) {
    llvm::PHINode *Phi = CreateEmptyPhi(var_name, mapType(var_name));
    write_local_variable(
        var_name, Phi);  // must write value first, or UpdatePhiOpdFromAllPredBB might be traped in endless recursion
    if (sealed) {
      // When sealed, we could read values from predecessors directly
      if (auto *PredBB = source_bb_->getSinglePredecessor()) {
        // if there is only one predecessor, we could bypass the created phi-node
        // UpdatePhiOpdFromAllPredBB is also able to do this optimization, it's just a hand-coded bypass
        auto v = resovle_manager_->GetResolver(PredBB).readLocalVariable(var_name);
        write_local_variable(var_name, v);
      } else {
        UpdatePhiOpdFromAllPredBB(var_name, Phi);
      }
    } else {
      // mark the phi node as imcomplete, and update it later
      incomplete_phis[Phi] = var_name;
    }
    return values_[var_name];
  }

  void UpdatePhiOpdFromAllPredBB(llvm::StringRef var_name, llvm::PHINode *Phi) {
    assert(sealed);  // only sealed bb can call this function
    for (auto I = llvm::pred_begin(source_bb_), E = llvm::pred_end(source_bb_); I != E; ++I) {
      // this call may cause recursion, for predecessor relation may create a loop
      auto value_from_i = resovle_manager_->GetResolver(*I).readLocalVariable(var_name);
      Phi->addIncoming(value_from_i, *I);
    }
    optimizePhi(Phi);
  }

  /**
   * 1. convert empty phi node to undef
   * 2. single value phi node to its value
   */
  void optimizePhi(llvm::PHINode *Phi) {
    // Check if all incoming values are same, if not, just return
    llvm::Value *Same = nullptr;
    for (llvm::Value *V : Phi->incoming_values()) {
      if (V == Same || V == Phi) continue;
      if (Same && V != Same) return;
      Same = V;
    }
    if (Same == nullptr) Same = llvm::UndefValue::get(Phi->getType());

    // replace all uses
    // If user of the phi is also a phi node ,that phi node may also be optimized to single value, so we get the
    // CandidatePhis first
    llvm::SmallVector<llvm::PHINode *, 8> CandidatePhis;
    for (llvm::Use &U : Phi->uses()) {
      if (auto *P = llvm::dyn_cast<llvm::PHINode>(U.getUser()))
        if (P != Phi) CandidatePhis.push_back(P);
    }
    Phi->replaceAllUsesWith(Same);
    Phi->eraseFromParent();
    for (auto *P : CandidatePhis) optimizePhi(P);
  }

  llvm::PHINode *CreateEmptyPhi(llvm::StringRef var_name, llvm::Type *type) {
    return source_bb_->empty() ? llvm::PHINode::Create(type, 0, "", source_bb_)
                               : llvm::PHINode::Create(type, 0, "", &source_bb_->front());
  }

  void Seal() {
    assert(!sealed && "Attempt to seal already sealed block");
    for (auto PhiDecl : incomplete_phis) {
      UpdatePhiOpdFromAllPredBB(PhiDecl.second, PhiDecl.first);
    }
    incomplete_phis.clear();
    sealed = true;
  }

 private:
  llvm::Type *mapType(llvm::StringRef var_name) {
    // todo: this function map var_name to it's type
    return llvm::Type::getDoubleTy(source_bb_->getContext());
  }
  BasicBlockSymResolverManager *const resovle_manager_;  // weak ref to resolve_mgr
  llvm::BasicBlock *const source_bb_;                    // weak ref to the source bb
  // Maps the variable (or formal parameter) to its definition.
  llvm::DenseMap<llvm::StringRef, llvm::TrackingVH<llvm::Value>> values_;
  // Set of incompleted phi instructions.
  llvm::DenseMap<llvm::PHINode *, llvm::StringRef> incomplete_phis;
  // Block is sealed, that is, all predecessors of the bb are added.
  bool sealed;
};

BasicBlockSymResolver &BasicBlockSymResolverManager::GetResolver(llvm::BasicBlock *bb) {
  if (map.count(bb) == 0) {
    map[bb] = std::make_shared<BasicBlockSymResolver>(this, bb);
  }
  return *map[bb];
}

class ASTToLLVM : public lox::ASTNodeVisitor<llvm::Value *> {
 public:
  explicit ASTToLLVM(llvm::LLVMContext &context) : context_(context), builder_(context) {
    double_ty_ = llvm::Type::getDoubleTy(context_);
    int32_t_ty_ = llvm::Type::getInt32Ty(context_);
  }

  std::unique_ptr<llvm::Module> Convert(lox::FunctionStmt *ast_module, const std::string &output_module_name) {
    ll_module_ = std::make_unique<llvm::Module>(output_module_name, context_);
    for (auto &stmt : ast_module->body) {
      NoValueVisit(stmt);
    }
    return std::move(ll_module_);
  }

  void Visit(AssignExpr *node) override {
    auto value = ValueVisit(node->value);
    bb_resolve_mgr.GetResolver(active_bb_).write_local_variable(node->attr->name->lexeme, value);
  }

  void Visit(LogicalExpr *node) override {}

  void Visit(BinaryExpr *node) override {
    switch (node->attr->op->type) {
      case TokenType::PLUS: {
        auto v0 = ValueVisit(node->left);
        auto v1 = ValueVisit(node->right);
        auto ret = llvm::BinaryOperator::Create(llvm::Instruction::FAdd, v0, v1, "add", active_bb_);
        VisitorReturn(ret);
      }
      default:
        throw LLVMTranslationError("not supported op");
    }
  }

  void Visit(GroupingExpr *node) override {}

  void Visit(LiteralExpr *node) override {
    switch (node->attr->value->type) {
      case TokenType::NUMBER:
        VisitorReturn(llvm::ConstantFP::get(double_ty_, std::stod(node->attr->value->lexeme)));
      case TokenType::STRING:
        VisitorReturn(llvm::ConstantDataArray::getString(context_, node->attr->value->lexeme, true));
      default:
        throw LLVMTranslationError("Not a valid Literal.");
    }
  }

  void Visit(UnaryExpr *node) override {}

  void Visit(CallExpr *node) override {}

  void Visit(GetAttrExpr *node) override {}

  void Visit(SetAttrExpr *node) override {}

  void Visit(VariableExpr *node) override {
    auto value = bb_resolve_mgr.GetResolver(active_bb_).readLocalVariable(node->attr->name->lexeme);
    VisitorReturn(value);
  }

  void Visit(CommaExpr *node) override {}

  void Visit(ListExpr *node) override {}

  void Visit(GetItemExpr *node) override {}

  void Visit(TensorExpr *node) override {}

  void Visit(VarDeclStmt *node) override {
    auto value = ValueVisit(node->initializer);
    bb_resolve_mgr.GetResolver(active_bb_).write_local_variable(node->attr->name->lexeme, value);
  }

  void Visit(WhileStmt *node) override {
    assert(function_hierarchy_.size() > 0);  // while is only allowed in function.
    auto fn = function_hierarchy_.back();
    llvm::BasicBlock *while_cond_bb = llvm::BasicBlock::Create(context_, "while.cond", fn);
    llvm::BasicBlock *while_body_bb = llvm::BasicBlock::Create(context_, "while.body", fn);
    llvm::BasicBlock *after_while_bb = llvm::BasicBlock::Create(context_, "after.while", fn);
    // branch to while_cond_bb
    builder_.CreateBr(while_cond_bb);

    // seal the current active bb
    bb_resolve_mgr.GetResolver(active_bb_).Seal();

    // switch to while_cond_bb
    SwitchBB(while_cond_bb);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, while_body_bb, after_while_bb);

    // switch to while_body_bb
    SwitchBB(while_body_bb);
    NoValueVisit(node->body);
    builder_.CreateBr(while_cond_bb);
    bb_resolve_mgr.GetResolver(while_cond_bb).Seal();
    bb_resolve_mgr.GetResolver(while_body_bb).Seal();

    // switch back
    SwitchBB(after_while_bb);
  }

  void Visit(ForStmt *node) override {}

  void Visit(ExprStmt *node) override { NoValueVisit(node->expression); }

  void Visit(FunctionStmt *node) override {
    // only a main function with no arguments is supported by now
    assert(node->attr->name->lexeme == "main");
    assert(!node->comma_expr_params || node->comma_expr_params->DynAs<CommaExpr>()->elements.empty());

    // create fn
    assert(node->attr->name->lexeme == "main");  // only main is supported by now
    llvm::FunctionType *fn_type = llvm::FunctionType::get(llvm::Type::getInt32Ty(context_), {}, false);
    llvm::Function *fn = llvm::Function::Create(fn_type, llvm::GlobalValue::ExternalLinkage, "main", ll_module_.get());

    // add new function to back as active function
    function_hierarchy_.push_back(fn);

    // create a guard to switch back to the current BB after the function
    llvm::IRBuilder<>::InsertPointGuard guard(builder_);
    // add entry bb and switch insert point to it
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(context_, "entry", fn);

    SwitchBB(BB);
    for (auto &stmt : node->body) {
      NoValueVisit(stmt);
    }
    // always inject a return 0 at the end of the function
    builder_.CreateRet(llvm::ConstantInt::get(int32_t_ty_, 0));
    // Builder.CreateRetVoid(); will be fine too

    // pop the function to disable it
    function_hierarchy_.pop_back();
    bb_resolve_mgr.GetResolver(active_bb_).Seal();
  }

  void Visit(ClassStmt *node) override {}

  void Visit(PrintStmt *node) override {}

  void Visit(ReturnStmt *node) override {
    assert(node->value);  // must return with value
    auto value = ValueVisit(node->value);
    assert(value->getType() == double_ty_);  // only number supported by now
    auto *cast = llvm::CastInst::Create(llvm::Instruction::FPToSI, value, int32_t_ty_, "", active_bb_);
    builder_.CreateRet(cast);
  }

  void Visit(BlockStmt *node) override {}

  void Visit(IfStmt *node) override {}

  void Visit(BreakStmt *node) override {}

 protected:
  void SwitchBB(llvm::BasicBlock *bb) {
    active_bb_ = bb;
    builder_.SetInsertPoint(active_bb_);
  }
  llvm::Value *readVariable(llvm::BasicBlock *BB, llvm::StringRef *var_name) {
    // we will generate 3 kinds of inst to read local, read global, read param
    // local is read through bb_resolve_mgr
    // global is through a builder_.CreateLoad(mapType(D), CGM.getGlobal(D))
    // param is treated as value on stack, which may also be a load instruction
    // closure is not support yet
    if (auto *V = llvm::dyn_cast<VariableDeclaration>(D)) {
      if (V->getEnclosingDecl() == Proc)
        return readLocalVariable(BB, var_name);
      else if (V->getEnclosingDecl() == CGM.getModuleDeclaration()) {
        return builder_.CreateLoad(mapType(D), CGM.getGlobal(D));
      } else
        llvm::report_fatal_error("Nested procedures not yet supported");
    } else if (auto *FP = llvm::dyn_cast<FormalParameterDeclaration>(D)) {
      if (FP->isVar()) {
        return builder_.CreateLoad(mapType(FP)->getPointerElementType(), FormalParams[FP]);
      } else
        return readLocalVariable(BB, D);
    } else
      llvm::report_fatal_error("Unsupported declaration");
  }

  llvm::Type *double_ty_;
  llvm::Type *int32_t_ty_;
  BasicBlockSymResolverManager bb_resolve_mgr;
  llvm::LLVMContext &context_;
  std::unique_ptr<llvm::Module> ll_module_;
  llvm::IRBuilder<> builder_;
  std::vector<llvm::Function *> function_hierarchy_;
  llvm::BasicBlock *active_bb_;
};

std::unique_ptr<llvm::Module> ConvertASTToLLVM(llvm::LLVMContext &context, lox::ASTNode *root) {
  auto ret = ASTToLLVM(context).Convert(root->DynAs<FunctionStmt>(), "main_module");
  ret->print(llvm::outs(), nullptr);  // todo: remove debug print later
  return ret;
}

}  // namespace lox::llvm_jit