#include "ast_to_llvm.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

#include <memory>

#include "lox/ast/ast.h"

namespace lox::llvm_jit {

struct BasicBlockDef {
  // Maps the variable (or formal parameter) to its definition.
  llvm::DenseMap<llvm::StringRef, llvm::TrackingVH<llvm::Value>> Defs;
  // Set of incompleted phi instructions.
  llvm::DenseMap<llvm::PHINode *, llvm::StringRef> IncompletePhis;
  // Block is sealed, that is, no more predecessors will be added.
  unsigned Sealed : 1;

  BasicBlockDef() : Sealed(0) {}
};

class ASTToLLVM : public lox::ASTNodeVisitor<llvm::Value *> {
 public:
  explicit ASTToLLVM(llvm::LLVMContext &context) : context_(context), builder_(context) {
    double_ty_ = llvm::Type::getDoubleTy(context_);
  }

  std::unique_ptr<llvm::Module> Convert(lox::FunctionStmt *ast_module, const std::string &output_module_name) {
    ll_module_ = std::make_unique<llvm::Module>(output_module_name, context_);
    for (auto &stmt : ast_module->body) {
      NoValueVisit(stmt);
    }
    return std::move(ll_module_);
  }

  void Visit(AssignExpr *node) override {}

  void Visit(LogicalExpr *node) override {}

  void Visit(BinaryExpr *node) override {}

  void Visit(GroupingExpr *node) override {}

  void Visit(LiteralExpr *node) override {}

  void Visit(UnaryExpr *node) override {}

  void Visit(CallExpr *node) override {}

  void Visit(GetAttrExpr *node) override {}

  void Visit(SetAttrExpr *node) override {}

  void Visit(VariableExpr *node) override {}

  void Visit(CommaExpr *node) override {}

  void Visit(ListExpr *node) override {}

  void Visit(GetItemExpr *node) override {}

  void Visit(TensorExpr *node) override {}

  void Visit(VarDeclStmt *node) override {}

  void Visit(WhileStmt *node) override {
    llvm::BasicBlock *while_cond_bb = llvm::BasicBlock::Create(context_, "while.cond");
    llvm::BasicBlock *while_body_bb = llvm::BasicBlock::Create(context_, "while.body");
    llvm::BasicBlock *after_while_bb = llvm::BasicBlock::Create(context_, "after.while");
    // branch to while_cond_bb
    builder_.CreateBr(while_cond_bb);

    // switch to while_cond_bb
    builder_.SetInsertPoint(while_cond_bb);
    llvm::Value *cond = ValueVisit(node->condition);
    builder_.CreateCondBr(cond, while_body_bb, after_while_bb);

    // switch to while_body_bb
    builder_.SetInsertPoint(while_body_bb);
    NoValueVisit(node->body);
    builder_.CreateBr(while_cond_bb);

    // switch back
    builder_.SetInsertPoint(after_while_bb);
  }

  void Visit(ForStmt *node) override {}

  void Visit(ExprStmt *node) override {}

  void Visit(FunctionStmt *node) override {
    // only a main function with no arguments is supported by now
    assert(node->attr->name->lexeme == "main");
    assert(node->comma_expr_params->DynAs<CommaExpr>()->elements.empty());

    // create a guard to switch back to the current BB after the function
    llvm::IRBuilder<>::InsertPointGuard guard(builder_);

    // create fn
    llvm::FunctionType *fn_type = llvm::FunctionType::get(llvm::Type::getInt32Ty(context_), {}, false);
    llvm::Function *fn = llvm::Function::Create(fn_type, llvm::GlobalValue::ExternalLinkage, "main", ll_module_.get());

    // add entry bb and switch insert point to it
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(context_, "entry", fn);

    builder_.SetInsertPoint(BB);
    for (auto &stmt : node->body) {
      NoValueVisit(stmt);
    }
    // always inject a return 0 at the end of the function
    builder_.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context_), 0));
  }

  void Visit(ClassStmt *node) override {}

  void Visit(PrintStmt *node) override {}

  void Visit(ReturnStmt *node) override {}

  void Visit(BlockStmt *node) override {}

  void Visit(IfStmt *node) override {}

  void Visit(BreakStmt *node) override {}

 protected:
  llvm::Type *double_ty_;

  llvm::DenseMap<llvm::BasicBlock *, BasicBlockDef> CurrentDef;

  llvm::LLVMContext &context_;
  std::unique_ptr<llvm::Module> ll_module_;
  llvm::IRBuilder<> builder_;
};

std::unique_ptr<llvm::Module> ConvertASTToLLVM(llvm::LLVMContext &context, lox::ASTNode *root) {
  auto ret = ASTToLLVM(context).Convert(root->DynAs<FunctionStmt>(), "main_module");
  ret->print(llvm::outs(), nullptr);  // todo: remove debug print later
  return ret;
}

}  // namespace lox::llvm_jit