import { requireNativeModule } from "expo";

import type { PnpNativeModule } from "./Pnp.types";

export default requireNativeModule<PnpNativeModule>("Pnp");
