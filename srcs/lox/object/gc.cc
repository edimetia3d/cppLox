//
// LICENSE: MIT
//

#include "lox/object/gc.h"

namespace lox {
GC &GC::Instance() {
  static GC obj;
  return obj;
}
void GC::collectGarbage() {
  markRoots();
  Sweep();
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
void GC::markRoots() {
  auto node = markers.begin();
  while (node != markers.end()) {
    node->marker_fn(node->marker_fn_arg);
    ++node;
  }
}
void GC::RegisterMarker(GC::MarkerFn fn, void *arg) { markers.insert(Marker{.marker_fn = fn, .marker_fn_arg = arg}); }
void GC::UnRegisterMarker(GC::MarkerFn fn, void *arg) { markers.erase(Marker{.marker_fn = fn, .marker_fn_arg = arg}); }

bool GC::Marker::operator<(const GC::Marker &rhs) const {
  return rhs.marker_fn_arg < marker_fn_arg && rhs.marker_fn < marker_fn;
}
GC::RegisterMarkerGuard::RegisterMarkerGuard(GC::MarkerFn fn, void *arg) : marker{fn, arg} {
  GC::Instance().RegisterMarker(marker.marker_fn, marker.marker_fn_arg);
}
GC::RegisterMarkerGuard::~RegisterMarkerGuard() {
  GC::Instance().UnRegisterMarker(marker.marker_fn, marker.marker_fn_arg);
}
}  // namespace lox