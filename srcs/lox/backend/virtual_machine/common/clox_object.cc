//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/common/clox_object.h"

#include <cstdio>
#include <cstring>

#include "lox/backend/virtual_machine/bytecode/chunk.h"

namespace lox {
namespace vm {
static uint32_t fnv_1a(uint8_t *data, int size) {
  uint32_t new_hash = 2166136261u;
  for (int i = 0; i < size; i++) {
    new_hash ^= data[i];
    new_hash *= 16777619;
  }
  return new_hash;
}
bool lox::vm::Obj::Equal(const lox::vm::Obj *rhs) const {
  return this == rhs;  // all string are interned, so we can compare directly
}
void Obj::Print() const {
#define q_printf(...)  \
  printf(__VA_ARGS__); \
  break
  switch (type) {
    case ObjType::OBJ_STRING:
      q_printf("%s", As<ObjInternedString>()->c_str());
    case ObjType::OBJ_FUNCTION:
      q_printf("<fn %s>", As<ObjFunction>()->name.c_str());
    case ObjType::OBJ_NATIVE_FUNCTION:
      q_printf("<native fn>");
    case ObjType::OBJ_RUNTIME_FUNCTION: {
      auto p = As<ObjRuntimeFunction>();
      if (p->isClosure()) {
        q_printf("<closure %s>", As<ObjRuntimeFunction>()->function->name.c_str());
      } else {
        q_printf("<fn %s>", As<ObjRuntimeFunction>()->function->name.c_str());
      }
    }
    case ObjType::OBJ_UPVALUE:
      q_printf("upvalue");
    default:
      q_printf("Unknown Obj type");
  }
#undef q_printf
}
LinkList<Obj *> &Obj::AllCreatedObj() {
  static LinkList<Obj *> ret;
  return ret;
}
uint32_t ObjInternedString::UpdateHash() {
  hash = fnv_1a((uint8_t *)c_str(), size());
  return hash;
}
ObjInternedString *ObjInternedString::Concat(const ObjInternedString *lhs, const ObjInternedString *rhs) {
  Buffer<char> buf;
  buf.reserve(lhs->size() + rhs->size() + 1);
  buf.push_buffer(lhs->c_str(), lhs->size());
  buf.push_buffer(rhs->c_str(), rhs->size());
  buf.push_back('\0');
  InternView view{.data = buf.data(), .hash = fnv_1a((uint8_t *)buf.data(), buf.size() - 1), .size = (buf.size() - 1)};
  auto entry = GetInternMap().Get(view);
  if (entry) {
    return entry->value;
  } else {
    return new ObjInternedString(std::move(buf));
  }
}
int ObjInternedString::size() const {
  return data.size() - 1;  // this is a c style str
}
ObjInternedString::ObjInternedString(const char *buf, int size) {
  data.push_buffer(buf, size);
  data.push_back('\0');
  UpdateHash();
  TryInternThis();
}
void ObjInternedString::TryInternThis() {
  if (!GetInternMap().Get(GetInternView())) {
    GetInternMap().Set(GetInternView(), this);
  } else {
    printf("Warning: Get repeated intern string\n");
  }
}
ObjInternedString::ObjInternedString(Buffer<char> &&buffer) {
  data = std::move(buffer);
  UpdateHash();
  TryInternThis();
}
ObjInternedString::InternMap &ObjInternedString::GetInternMap() {
  static InternMap interning_map(32);
  return interning_map;
}
ObjInternedString *ObjInternedString::Make(const char *data, int size) {
  InternView view{.data = data, .hash = fnv_1a((uint8_t *)data, size), .size = size};
  auto entry = GetInternMap().Get(view);
  if (entry) {
    return entry->value;
  } else {
    return new ObjInternedString(data, size);
  }
}
ObjInternedString::~ObjInternedString() { GetInternMap().Del(GetInternView()); }

void Obj::Destroy(Obj *obj) {
  switch (obj->type) {
    case ObjType::OBJ_STRING:
      delete obj->As<ObjInternedString>();
      break;
    case ObjType::OBJ_FUNCTION:
      delete obj->As<ObjFunction>();
      break;
    case ObjType::OBJ_NATIVE_FUNCTION:
      delete obj->As<ObjNativeFunction>();
      break;
    case ObjType::OBJ_RUNTIME_FUNCTION:
      delete obj->As<ObjRuntimeFunction>();
      break;
    case ObjType::OBJ_UPVALUE:
      delete obj->As<ObjUpvalue>();
      break;
    default:
      printf("Destroying Unknown Type.\n");
  }
}

ObjFunction::ObjFunction() {
  chunk = new Chunk();
}
ObjFunction::~ObjFunction() { delete chunk; }
}  // namespace vm
}  // namespace lox
