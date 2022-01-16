//
// LICENSE: MIT
//
#define SPDLOG_EOL ""

#include <spdlog/spdlog.h>

#include "lox/backend/virtual_machine/core/chunk.h"
#include "lox/backend/virtual_machine/core/vm.h"
#include "lox/backend/virtual_machine/object/object.h"
#include "lox/object/value.h"

namespace lox::vm {

static void PrintValue(Value v) {
  switch (v.Type()) {
    case ValueType::BOOL: {
      SPDLOG_DEBUG(v.AsBool() ? "true" : "false");
      break;
    }
    case ValueType::NUMBER: {
      SPDLOG_DEBUG("%f", v.AsNumber());
      break;
    }
    case ValueType::NIL: {
      SPDLOG_DEBUG("nil");
      break;
    }
    case ValueType::OBJECT: {
      SPDLOG_DEBUG("%s", v.AsObject()->Str().c_str());
      break;
    }
    default: {
      SPDLOG_DEBUG("UnKownValue");
    }
  }
}

static int SimpleInstruction(const char *name, int offset) {
  SPDLOG_DEBUG("%s\n", name);
  return offset + 1;
}

static int ConstantInstruction(const char *name, const Chunk *chunk, int offset) {
  uint8_t const_index = chunk->code[offset + 1];
  SPDLOG_DEBUG("%-16s %4d '", name, const_index);
  PrintValue(chunk->constants[const_index]);
  SPDLOG_DEBUG("'\n");
  return offset + 2;
}

static int ByteInstruction(const char *name, const Chunk *chunk, int offset) {
  uint8_t slot = chunk->code[offset + 1];
  SPDLOG_DEBUG("%-16s %4d\n", name, slot);
  return offset + 2;
}

static int JumpInstruction(const char *name, int sign, const Chunk *chunk, int offset) {
  uint16_t jump = (uint16_t)(chunk->code[offset + 1] << 8);
  jump |= chunk->code[offset + 2];
  SPDLOG_DEBUG("%-16s %4d -> %d\n", name, offset, offset + 3 + sign * jump);
  return offset + 3;
}

static int InvokeInstruction(const char *op_name, const Chunk *chunk, int offset) {
  uint8_t constant = chunk->code[offset + 1];
  uint8_t argCount = chunk->code[offset + 2];
  SPDLOG_DEBUG("%-16s (%d args) %4d '", op_name, argCount, constant);
  PrintValue(chunk->constants[constant]);
  SPDLOG_DEBUG("'\n");
  return offset + 3;
}

int DumpInstruction(const Chunk *chunk, int offset) {
  SPDLOG_DEBUG("%04d ", offset);
  if (offset > 0 && chunk->lines[offset] == chunk->lines[offset - 1]) {
    SPDLOG_DEBUG("   | ");
  } else {
    SPDLOG_DEBUG("%4d ", chunk->lines[offset]);
  }
  auto instruction = static_cast<OpCode>(chunk->code[offset]);
  switch (instruction) {
    case OpCode::OP_CONSTANT:
      return ConstantInstruction("OP_CONSTANT", chunk, offset);
    case OpCode::OP_NIL:
      return SimpleInstruction("OP_NIL", offset);
    case OpCode::OP_TRUE:
      return SimpleInstruction("OP_TRUE", offset);
    case OpCode::OP_FALSE:
      return SimpleInstruction("OP_FALSE", offset);
    case OpCode::OP_EQUAL:
      return SimpleInstruction("OP_EQUAL", offset);
    case OpCode::OP_GREATER:
      return SimpleInstruction("OP_GREATER", offset);
    case OpCode::OP_LESS:
      return SimpleInstruction("OP_LESS", offset);
    case OpCode::OP_ADD:
      return SimpleInstruction("OP_ADD", offset);
    case OpCode::OP_SUBTRACT:
      return SimpleInstruction("OP_SUBTRACT", offset);
    case OpCode::OP_MULTIPLY:
      return SimpleInstruction("OP_MULTIPLY", offset);
    case OpCode::OP_DIVIDE:
      return SimpleInstruction("OP_DIVIDE", offset);
    case OpCode::OP_NOT:
      return SimpleInstruction("OP_NOT", offset);
    case OpCode::OP_NEGATE:
      return SimpleInstruction("OP_NEGATE", offset);
    case OpCode::OP_RETURN:
      return SimpleInstruction("OP_RETURN", offset);
    case OpCode::OP_PRINT:
      return SimpleInstruction("OP_PRINT", offset);
    case OpCode::OP_POP:
      return SimpleInstruction("OP_POP", offset);
    case OpCode::OP_DEFINE_GLOBAL:
      return ConstantInstruction("OP_DEFINE_GLOBAL", chunk, offset);
    case OpCode::OP_GET_GLOBAL:
      return ConstantInstruction("OP_GET_GLOBAL", chunk, offset);
    case OpCode::OP_SET_GLOBAL:
      return ConstantInstruction("OP_SET_GLOBAL", chunk, offset);
    case OpCode::OP_GET_LOCAL:
      return ByteInstruction("OP_GET_LOCAL", chunk, offset);
    case OpCode::OP_SET_LOCAL:
      return ByteInstruction("OP_SET_LOCAL", chunk, offset);
    case OpCode::OP_JUMP:
      return JumpInstruction("OP_JUMP", 1, chunk, offset);
    case OpCode::OP_JUMP_IF_FALSE:
      return JumpInstruction("OP_JUMP_IF_FALSE", 1, chunk, offset);
    case OpCode::OP_JUMP_BACK:
      return JumpInstruction("OP_JUMP_BACK", -1, chunk, offset);
    case OpCode::OP_CALL:
      return ByteInstruction("OP_CALL", chunk, offset);
    case OpCode::OP_CLOSURE: {
      offset++;
      uint8_t constant = chunk->code[offset++];
      SPDLOG_DEBUG("%-16s %4d ", "OP_CLOSURE", constant);
      chunk->constants[constant].PrintLn();
      auto function = chunk->constants[constant].AsObject()->DynAs<ObjFunction>();
      int upvalue_count = chunk->code[offset++];
      for (int j = 0; j < upvalue_count; j++) {
        int isLocal = chunk->code[offset++];
        int index = chunk->code[offset++];
        SPDLOG_DEBUG("%04d      |                     %s %d\n", offset - 2, isLocal ? "local" : "upvalue", index);
      }
      return offset;
    }
    case OpCode::OP_GET_UPVALUE:
      return ByteInstruction("OP_GET_UPVALUE", chunk, offset);
    case OpCode::OP_SET_UPVALUE:
      return ByteInstruction("OP_SET_UPVALUE", chunk, offset);
    case OpCode::OP_CLASS:
      return ConstantInstruction("OP_CLASS", chunk, offset);
    case OpCode::OP_GET_ATTR:
      return ConstantInstruction("OP_GET_ATTR", chunk, offset);
    case OpCode::OP_SET_ATTR:
      return ConstantInstruction("OP_SET_ATTR", chunk, offset);
    case OpCode::OP_METHOD:
      return ConstantInstruction("OP_METHOD", chunk, offset);
    case OpCode::OP_INVOKE:
      return InvokeInstruction("OP_INVOKE", chunk, offset);
    case OpCode::OP_INHERIT:
      return SimpleInstruction("OP_INHERIT", offset);
    default:
      SPDLOG_DEBUG("Unknown opcode %d\n", (int)instruction);
      return offset + 1;
  }
}

void DumpChunkCode(const Chunk *chunk) {
  // DumpInstruction will return the offset of the next instruction.
  // For some instruction has a variable length.
  for (int offset = 0; offset < chunk->ChunkSize();) {
    offset = DumpInstruction(chunk, offset);
  }
}

void DumpChunkConstant(const Chunk *chunk) {
  for (const Value *p = chunk->constants.data(); p != chunk->constants.data() + chunk->constants.size(); ++p) {
    SPDLOG_DEBUG("[ ");
    PrintValue(*p);
    SPDLOG_DEBUG(" ]");
  }
  SPDLOG_DEBUG("\n");
}
void DumpStack(const VM *vm) {
  SPDLOG_DEBUG("Stack:");
  for (const Value *slot = vm->stack_; slot != vm->sp_; ++slot) {
    SPDLOG_DEBUG("[ ");
    PrintValue(*slot);
    SPDLOG_DEBUG(" ]");
  }
  SPDLOG_DEBUG("\n");
}
void DumpGlobal(const VM *vm) {
  SPDLOG_DEBUG("Globals:");
  auto iter = vm->globals_.begin();
  while (iter != vm->globals_.end()) {
    SPDLOG_DEBUG("{ ");
    SPDLOG_DEBUG("%s", iter->first->c_str());
    SPDLOG_DEBUG(" : ");
    PrintValue(iter->second);
    SPDLOG_DEBUG(" }");
    ++iter;
  }
  SPDLOG_DEBUG("\n");
}
}  // namespace lox::vm
