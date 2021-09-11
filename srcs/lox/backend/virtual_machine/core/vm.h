//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_VM_VM_H_
#define CLOX_SRCS_CLOX_VM_VM_H_
#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/common/err_code.h"
#define VM_STACK_MAX 256
namespace lox {
namespace vm {

/**
 * A stack machine
 */
class VM {
 public:
  static VM *Instance();
  ErrCode Interpret(Chunk *chunk);

 private:
  void ResetStack();
  VM() { ResetStack(); }
  ErrCode Run();
  void Push(Value value);
  Value Peek(int distance = 0);
  Value Pop();
  Chunk *chunk_;
  uint8_t *ip_;
  Value stack_[VM_STACK_MAX];
  HashMap<ObjInternedString *, Value, ObjInternedString::Hash> globals_;
  Value *sp_ = nullptr;
  void DumpStack() const;
  void DumpGlobals();
  void runtimeError(const char *format, ...);
  ~VM();
};
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_VM_VM_H_
