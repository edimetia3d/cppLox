//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_VM_VM_H_
#define CLOX_SRCS_CLOX_VM_VM_H_
#include "lox/backend/virtual_machine/bytecode/chunk.h"

#define VM_STACK_MAX 256
namespace lox {
namespace vm {
enum class InterpretResult { INTERPRET_OK, INTERPRET_COMPILE_ERROR, INTERPRET_RUNTIME_ERROR };

class VM {
 public:
  static VM *Instance();
  InterpretResult Interpret(Chunk *chunk);

 private:
  void ResetStack();
  VM() { ResetStack(); }
  InterpretResult Run();
  void Push(Value value);
  Value Pop();
  Chunk *chunk_;
  uint8_t *ip_;
  Value stack_[VM_STACK_MAX];
  Value *sp_ = nullptr;
};
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_VM_VM_H_
