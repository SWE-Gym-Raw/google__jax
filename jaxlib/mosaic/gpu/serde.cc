/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/mosaic/gpu/serde.h"

#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/include/llvm/ADT/StringMap.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mosaic::gpu {

namespace {

constexpr std::string_view kMangledDialect = "stable_mosaic_gpu.";
constexpr mlir::StringRef kVersionAttrName = "stable_mosaic_gpu.version";
// When this is bumped, we should file a TODO to update the forward-compatible
// version in Mosaic GPU lowering in a month!
constexpr int kVersion = 1;

mlir::StringRef mangle(mlir::StringRef name, std::string* storage) {
  storage->clear();
  storage->reserve(kMangledDialect.size() + name.size());
  storage->insert(storage->end(), kMangledDialect.begin(),
                  kMangledDialect.end());
  storage->insert(storage->end(), name.begin(), name.end());
  return *storage;
}

std::optional<mlir::StringRef> demangle(mlir::StringRef name) {
  if (!name.starts_with(kMangledDialect)) {
    return std::nullopt;
  }
  return name.drop_front(kMangledDialect.size());
}

using rule_type = std::function<mlir::LogicalResult(mlir::Operation*, int)>;

const llvm::StringMap<rule_type>& upgrade_rules() {
  static auto rules = new llvm::StringMap<rule_type>{};
  return *rules;
}

const llvm::StringMap<rule_type>& downgrade_rules() {
  static auto rules = new llvm::StringMap<rule_type>{};
  return *rules;
}

}  // namespace

void SerdePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  if (!serialize.hasValue()) {
    module.emitError("serialize option must be specified");
    return signalPassFailure();
  }
  int serialize_version = target_version.hasValue() ? target_version : kVersion;
  if (serialize && serialize_version > kVersion) {
    module.emitError("The highest supported version is ")
        << kVersion << " but requested serialization at version "
        << serialize_version;
    return signalPassFailure();
  }
  if (serialize && !module->getContext()->allowsUnregisteredDialects()) {
    module.emitError() << "Cannot serialize within a context that does not "
                          "allow unregistered dialects.";
    signalPassFailure();
    return;
  }
  int version = kVersion;
  if (serialize) {
    module->setAttr(
        kVersionAttrName,
        mlir::IntegerAttr::get(mlir::IntegerType::get(module->getContext(), 64),
                               serialize_version));
  } else {
    mlir::IntegerAttr version_attr =
        module->getAttrOfType<mlir::IntegerAttr>(kVersionAttrName);
    if (!version_attr) {
      module->emitError("Missing or invalid Mosaic GPU version attribute");
      signalPassFailure();
      return;
    }
    if (version_attr.getInt() > kVersion) {
      module->emitError("Unsupported Mosaic GPU version:  expected <= ")
          << kVersion << " but got " << version_attr.getInt();
      signalPassFailure();
      return;
    }
    version = version_attr.getInt();
    module->removeAttr(kVersionAttrName);
  }
  std::string name_storage;
  auto result = module.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ModuleOp>(op)) {  // Don't mangle the ModuleOp itself.
      return mlir::WalkResult::advance();
    }
    std::optional<mlir::OperationName> new_name;
    if (serialize) {
      auto new_name_str = mangle(op->getName().getStringRef(), &name_storage);
      new_name = mlir::OperationName(new_name_str, op->getContext());
    } else {
      if (auto demangled = demangle(op->getName().getStringRef())) {
        auto new_name_str = *demangled;
        if (auto registered = mlir::RegisteredOperationName::lookup(
                new_name_str, op->getContext())) {
          new_name = *registered;
        } else {
          new_name = mlir::OperationName(new_name_str, op->getContext());
        }
      } else {
        op->emitError("Operation not in a serialized form");
        return mlir::WalkResult::interrupt();
      }
      // Upgrade the op to the current version, if needed.
      if (const auto rule = upgrade_rules().find(new_name->getStringRef());
          rule != upgrade_rules().end()) {
        if (rule->second(op, version).failed()) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
    auto new_op = mlir::Operation::create(
        op->getLoc(), *new_name, op->getResultTypes(), op->getOperands(),
        op->getAttrs(), nullptr, op->getSuccessors(), op->getRegions());
    // Downgrade the op to the target version, if needed.
    if (serialize && kVersion != serialize_version) {
      if (const auto rule =
              downgrade_rules().find(op->getName().getStringRef());
          rule != downgrade_rules().end()) {
        if (rule->second(new_op, serialize_version).failed()) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
    op->getBlock()->getOperations().insertAfter(mlir::Block::iterator(op),
                                                new_op);
    op->replaceAllUsesWith(new_op->getResults());
    op->erase();
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace mosaic::gpu
