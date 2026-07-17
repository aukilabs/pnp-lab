import type { PnpNativeModule } from "./Pnp.types";

const unavailable = async (): Promise<never> => {
  throw new Error("expo-pnp is native-only; use pnp-wasm on web.");
};

export default {
  solvePnpCameraPose: unavailable,
  estimateSquarePoseFromRays: unavailable,
} satisfies PnpNativeModule;
