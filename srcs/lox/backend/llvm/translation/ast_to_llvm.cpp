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

struct BasicBlockDef {
  // Maps the variable (or formal parameter) to its definition.
  llvm::DenseMap<llvm::StringRef, llvm::TrackingVH<llvm::Value>> defs;
  // Set of incompleted phi instructions.
  llvm::DenseMap<llvm::PHINode *, llvm::StringRef> incomplete_phis;
  // Block is sealed, that is, all predecessors of the bb are added.
  bool sealed;

  BasicBlockDef() : sealed(false) {}
};

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
    write_local_variable(active_bb_, node->attr->name->lexeme, value);
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
      default:
        throw LLVMTranslationError("Not a valid Literal.");
    }
  }

  void Visit(UnaryExpr *node) override {}

  void Visit(CallExpr *node) override {}

  void Visit(GetAttrExpr *node) override {}

  void Visit(SetAttrExpr *node) override {}

  void Visit(VariableExpr *node) override { VisitorReturn(readLocalVariable(active_bb_, node->attr->name->lexeme)); }

  void Visit(CommaExpr *node) override {}

  void Visit(ListExpr *node) override {}

  void Visit(GetItemExpr *node) override {}

  void Visit(TensorExpr *node) override {}

  void Visit(VarDeclStmt *node) override {
    auto value = ValueVisit(node->initializer);
    write_local_variable(active_bb_, node->attr->name->lexeme, value);
  }

  void Visit(WhileStmt *node) override {
    assert(function_hierarchy_.size() > 0);  // while is only allowed in function.
    auto fn = function_hierarchy_.back();
    llvm::BasicBlock *while_cond_bb = llvm::BasicBlock::Create(context_, "while.cond", fn);
    llvm::BasicBlock *while_body_bb = llvm::BasicBlock::Create(context_, "while.body", fn);
    llvm::BasicBlock *after_while_bb = llvm::BasicBlock::Create(context_, "after.while", fn);
    // branch to while_cond_bb
    builder_.CreateBr(while_cond_bb);

    // switch to while_cond_bb
    SwitchBB(while_cond_bb);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, while_body_bb, after_while_bb);

    // switch to while_body_bb
    SwitchBB(while_body_bb);
    NoValueVisit(node->body);
    builder_.CreateBr(while_cond_bb);
    sealBlock(while_body_bb);
    sealBlock(while_cond_bb);

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

    // pop the function to disable it
    function_hierarchy_.pop_back();
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
  llvm::Type *double_ty_;
  llvm::Type *int32_t_ty_;

  llvm::DenseMap<llvm::BasicBlock *, BasicBlockDef> current_defs_;

  void write_local_variable(llvm::BasicBlock *BB, llvm::StringRef var_name, llvm::Value *value) {
    current_defs_[BB].defs[var_name] = value;
  }
  llvm::Value *readLocalVariable(llvm::BasicBlock *BB, llvm::StringRef var_name) {
    auto Val = current_defs_[BB].defs.find(var_name);
    if (Val != current_defs_[BB].defs.end()) return Val->second;
    // when code goes here, a variable could only be found in other bb, so a phi will be needed.
    return read_var_from_predecessor(BB, var_name);
  }
  llvm::Value *read_var_from_predecessor(llvm::BasicBlock *BB, llvm::StringRef var_name) {
    llvm::Value *Val = nullptr;
    if (!current_defs_[BB].sealed) {
      // it is possible that not all predecessor is known
      llvm::PHINode *Phi = CreateEmptyPhi(BB, var_name);
      current_defs_[BB].incomplete_phis[Phi] = var_name;
      Val = Phi;
    } else if (auto *PredBB = BB->getSinglePredecessor()) {
      Val = readLocalVariable(PredBB, var_name);
    } else {
      llvm::PHINode *Phi = CreateEmptyPhi(BB, var_name);
      Val = Phi;
      write_local_variable(BB, var_name, Val);
      addPhiOperands(BB, var_name, Phi);
    }
    write_local_variable(BB, var_name, Val);
    return Val;
  }

  void addPhiOperands(llvm::BasicBlock *BB, llvm::StringRef var_name, llvm::PHINode *Phi) {
    for (auto I = llvm::pred_begin(BB), E = llvm::pred_end(BB); I != E; ++I) {
      Phi->addIncoming(readLocalVariable(*I, var_name), *I);
    }
    optimizePhi(Phi);
  }

  void optimizePhi(llvm::PHINode *Phi) {
    llvm::Value *Same = nullptr;
    for (llvm::Value *V : Phi->incoming_values()) {
      if (V == Same || V == Phi) continue;
      if (Same && V != Same) return;
      Same = V;
    }
    if (Same == nullptr) Same = llvm::UndefValue::get(Phi->getType());
    // Collect phi instructions using this one.
    llvm::SmallVector<llvm::PHINode *, 8> CandidatePhis;
    for (llvm::Use &U : Phi->uses()) {
      if (auto *P = llvm::dyn_cast<llvm::PHINode>(U.getUser()))
        if (P != Phi) CandidatePhis.push_back(P);
    }
    Phi->replaceAllUsesWith(Same);
    Phi->eraseFromParent();
    for (auto *P : CandidatePhis) optimizePhi(P);
  }

  llvm::PHINode *CreateEmptyPhi(llvm::BasicBlock *BB, llvm::StringRef var_name) {
    return BB->empty() ? llvm::PHINode::Create(mapType(var_name), 0, "", BB)
                       : llvm::PHINode::Create(mapType(var_name), 0, "", &BB->front());
  }

  llvm::Type *mapType(llvm::StringRef var_name) {
    // todo: this function should get the type of var_name
    return double_ty_;
  }

  llvm::LLVMContext &context_;
  std::unique_ptr<llvm::Module> ll_module_;
  llvm::IRBuilder<> builder_;
  std::vector<llvm::Function *> function_hierarchy_;
  llvm::BasicBlock *active_bb_;
  void SwitchBB(llvm::BasicBlock *bb) {
    active_bb_ = bb;
    builder_.SetInsertPoint(active_bb_);
  }
  void sealBlock(llvm::BasicBlock *BB) {
    // a seal should usually be called before some switchBB
    assert(!current_defs_[BB].sealed && "Attempt to seal already sealed block");
    for (auto PhiDecl : current_defs_[BB].incomplete_phis) {
      addPhiOperands(BB, PhiDecl.second, PhiDecl.first);
    }
    current_defs_[BB].incomplete_phis.clear();
    current_defs_[BB].sealed = true;
  }
};

std::unique_ptr<llvm::Module> ConvertASTToLLVM(llvm::LLVMContext &context, lox::ASTNode *root) {
  auto ret = ASTToLLVM(context).Convert(root->DynAs<FunctionStmt>(), "main_module");
  ret->print(llvm::outs(), nullptr);  // todo: remove debug print later
  return ret;
}

}  // namespace lox::llvm_jit