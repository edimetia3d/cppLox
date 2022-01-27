//
// LICENSE: MIT
//

#include "lox/object/gc.h"

#include <spdlog/spdlog.h>

namespace lox {
GC &GC::Instance() {
  static GC obj;
  return obj;
}

void GC::Collect() {
  MarkRoots();
  Sweep();
  gc_threashold = Object::AllCreatedObj().size() * 1.2;
}

void GC::RecursiveMark(Object *object) {
  if (object == nullptr || marked.contains(object)) return;
  marked[object] = true;
  for (auto ref : object->References()) {
    RecursiveMark(ref);
  }
}

void GC::Sweep() {
  auto &list = Object::AllCreatedObj();
  auto iter = list.begin();
  std::set<Object *> to_del;
  while (iter != list.end()) {
    if (!marked.contains(*iter)) {
      to_del.insert((*iter));
    }
    ++iter;
  }
  iter = to_del.begin();
  while (iter != to_del.end()) {
    delete (*iter);
    ++iter;
  }
  marked.clear();
}
void GC::MarkRoots() {
  for (auto pair : markers) {
    pair.second();
  }
}
bool GC::TryCollet() {
  if (Object::AllCreatedObj().size() > gc_threashold) {
    Collect();
    return true;
  }
  return false;
}
int GC::ForceClearAll() {
  int count = 0;
  auto objs = Object::AllCreatedObj();
  for (auto key : objs) {
    ++count;
    delete key;
  }
  return count;
}

}  // namespace lox