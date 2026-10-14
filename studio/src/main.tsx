import { StrictMode, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { api, clearSession } from "./lib/api";
import { StoreProvider } from "./lib/store";
import type { Me } from "./lib/types";
import AuthScreen from "./views/AuthScreen";

function AuthGate() {
  const [me, setMe] = useState<Me | null | undefined>(undefined);
  const refreshMe = useCallback(() => api.me().then(setMe, () => setMe(null)), []);
  useEffect(() => {
    refreshMe();
    const onUnauthorized = () => {
      clearSession();
      setMe(null);
    };
    window.addEventListener("infinity:unauthorized", onUnauthorized);
    return () => window.removeEventListener("infinity:unauthorized", onUnauthorized);
  }, [refreshMe]);
  if (me === undefined) return <div className="h-full bg-bg" aria-busy="true" />;
  if (me === null) return <AuthScreen onAuthed={refreshMe} />;
  return (
    <StoreProvider key={me.user.id}>
      <App me={me} refreshMe={refreshMe} onLoggedOut={() => setMe(null)} />
    </StoreProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AuthGate />
  </StrictMode>,
);
