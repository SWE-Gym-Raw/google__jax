/* Copyright 2024 The JAX Authors.

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

#include "jaxlib/mosaic/dialect/tpu_ext/integrations/c/tpu_ext_dialect.h"

#include "mlir/CAPI/Registration.h"
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_dialect.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TPUExt, tpu_ext,
                                      mlir::tpu::ext::TPUExtDialect);
}
