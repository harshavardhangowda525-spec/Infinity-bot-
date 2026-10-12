import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { CoreState } from "../lib/types";

/** Lemniscate of Bernoulli with a small z-lift so the two lobes pass over each other at the crossing. */
class Lemniscate extends THREE.Curve<THREE.Vector3> {
  constructor(private a = 1.6, private lift = 0.22) {
    super();
  }
  getPoint(t: number, target = new THREE.Vector3()) {
    const ang = t * Math.PI * 2;
    const s = Math.sin(ang);
    const c = Math.cos(ang);
    const d = 1 + s * s;
    return target.set((this.a * c) / d, (this.a * s * c) / d, this.lift * s);
  }
}

/** A ribbon that winds around the base lemniscate (several of these intertwine). */
class WindingCurve extends THREE.Curve<THREE.Vector3> {
  private tmpA = new THREE.Vector3();
  private tmpB = new THREE.Vector3();
  constructor(private base: Lemniscate, private radius: number, private turns: number, private phase: number) {
    super();
  }
  getPoint(t: number, target = new THREE.Vector3()) {
    const p = this.base.getPoint(t, target);
    const q = this.base.getPoint((t + 0.001) % 1, this.tmpA);
    const tangent = this.tmpB.subVectors(q, p).normalize();
    const perp = new THREE.Vector3(-tangent.y, tangent.x, 0).normalize();
    const ang = t * Math.PI * 2 * this.turns + this.phase;
    return p.addScaledVector(perp, Math.cos(ang) * this.radius).add(new THREE.Vector3(0, 0, Math.sin(ang) * this.radius));
  }
}

const PALETTE: Record<CoreState, { color: string; emissive: string; intensity: number; spin: number; flow: number }> = {
  idle: { color: "#7fe9ff", emissive: "#1d6cff", intensity: 0.25, spin: 0.08, flow: 0.02 },
  listening: { color: "#8ff1ff", emissive: "#2aa8ff", intensity: 0.45, spin: 0.1, flow: 0.04 },
  thinking: { color: "#8fd8ff", emissive: "#5a5cff", intensity: 0.55, spin: 0.35, flow: 0.12 },
  executing: { color: "#7fe9ff", emissive: "#20b8ff", intensity: 0.7, spin: 0.22, flow: 0.2 },
  speaking: { color: "#a9f3ff", emissive: "#3a8bff", intensity: 0.5, spin: 0.1, flow: 0.05 },
  success: { color: "#9ff7de", emissive: "#1fd6a0", intensity: 0.55, spin: 0.05, flow: 0.03 },
  error: { color: "#ffd59a", emissive: "#ff7a2f", intensity: 0.45, spin: 0.03, flow: 0.01 },
};

export interface CoreProps {
  state: CoreState;
  amplitude: React.MutableRefObject<number>;
  lowPower: boolean;
  reducedMotion: boolean;
}

function Hologram({ state, amplitude, lowPower, reducedMotion }: CoreProps) {
  const group = useRef<THREE.Group>(null);
  const particles = useRef<THREE.Points>(null);
  const pulse = useRef<THREE.Mesh>(null);
  const mats = useRef<THREE.MeshPhysicalMaterial[]>([]);
  const base = useMemo(() => new Lemniscate(), []);
  const ribbons = useMemo(() => {
    const defs = lowPower
      ? [{ r: 0.0, turns: 0, phase: 0, tube: 0.11 }, { r: 0.12, turns: 3, phase: 0, tube: 0.035 }]
      : [
          { r: 0.0, turns: 0, phase: 0, tube: 0.1 },
          { r: 0.13, turns: 3, phase: 0, tube: 0.034 },
          { r: 0.13, turns: 3, phase: (Math.PI * 2) / 3, tube: 0.03 },
          { r: 0.13, turns: 3, phase: (Math.PI * 4) / 3, tube: 0.026 },
        ];
    return defs.map((d) =>
      new THREE.TubeGeometry(d.r ? new WindingCurve(base, d.r, d.turns + 1, d.phase) : base, lowPower ? 220 : 420, d.tube, lowPower ? 8 : 14, true),
    );
  }, [base, lowPower]);

  const halo = useMemo(() => new THREE.TubeGeometry(base, lowPower ? 160 : 300, 0.2, lowPower ? 8 : 12, true), [base, lowPower]);
  const haloMat = useRef<THREE.MeshBasicMaterial>(null);

  const count = lowPower ? 260 : 900;
  const { positions, offsets, jitter } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const offsets = new Float32Array(count);
    const jitter = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      offsets[i] = Math.random();
      jitter[i * 3] = (Math.random() - 0.5) * 0.35;
      jitter[i * 3 + 1] = (Math.random() - 0.5) * 0.35;
      jitter[i * 3 + 2] = (Math.random() - 0.5) * 0.35;
    }
    return { positions, offsets, jitter };
  }, [count]);

  const visual = useRef({ ...PALETTE.idle, scale: 1, flash: 0 });
  const prevState = useRef<CoreState>(state);
  const tmp = useMemo(() => new THREE.Vector3(), []);
  const targetColor = useMemo(() => new THREE.Color(), []);
  const targetEmissive = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    if (state === "success" && prevState.current !== "success") visual.current.flash = 1;
    prevState.current = state;
  }, [state]);

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.05);
    const target = PALETTE[state];
    const v = visual.current;
    const k = 1 - Math.exp(-dt * 3);
    v.spin += (target.spin - v.spin) * k;
    v.flow += (target.flow - v.flow) * k;
    v.intensity += (target.intensity - v.intensity) * k;
    v.flash = Math.max(0, v.flash - dt * 0.8);

    const amp = state === "listening" || state === "speaking" ? amplitude.current : 0;
    const breathe = reducedMotion ? 0 : Math.sin(performance.now() / 1400) * 0.012;
    const targetScale = 1 + breathe + amp * 0.12 + v.flash * 0.06;
    v.scale += (targetScale - v.scale) * (1 - Math.exp(-dt * 10));

    if (group.current) {
      if (!reducedMotion) {
        group.current.rotation.y += v.spin * dt;
        group.current.rotation.x = Math.sin(performance.now() / 6000) * 0.18 - 0.25;
      }
      group.current.scale.setScalar(v.scale);
    }
    targetColor.set(target.color);
    targetEmissive.set(target.emissive);
    for (const m of mats.current) {
      if (!m) continue;
      m.color.lerp(targetColor, k);
      m.emissive.lerp(targetEmissive, k);
      m.emissiveIntensity = v.intensity * (0.8 + amp * 0.8) + v.flash * 0.6;
    }

    if (haloMat.current) {
      haloMat.current.color.lerp(targetEmissive, k);
      haloMat.current.opacity = 0.05 + v.intensity * 0.12 + amp * 0.12 + v.flash * 0.15;
    }

    if (particles.current && !reducedMotion) {
      const attr = particles.current.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < count; i++) {
        offsets[i] = (offsets[i] + v.flow * dt * (0.6 + (i % 7) * 0.08)) % 1;
        base.getPoint(offsets[i], tmp);
        const spread = 1 + amp * 1.5;
        attr.setXYZ(i, tmp.x + jitter[i * 3] * spread, tmp.y + jitter[i * 3 + 1] * spread, tmp.z + jitter[i * 3 + 2] * spread);
      }
      attr.needsUpdate = true;
    }

    if (pulse.current) {
      const show = state === "executing";
      pulse.current.visible = show;
      if (show) {
        const t = (performance.now() / 1800) % 1;
        base.getPoint(t, tmp);
        pulse.current.position.copy(tmp);
      }
    }
  });

  return (
    <group ref={group}>
      {ribbons.map((geo, i) => (
        <mesh key={i} geometry={geo}>
          {lowPower ? (
            <meshPhysicalMaterial
              ref={(m) => {
                mats.current[i] = m as THREE.MeshPhysicalMaterial;
              }}
              color="#7fe9ff"
              emissive="#1d6cff"
              transparent
              opacity={i === 0 ? 0.42 : 0.85}
              roughness={0.18}
              metalness={0.05}
              clearcoat={1}
              clearcoatRoughness={0.1}
              depthWrite={i !== 0}
            />
          ) : (
            <meshPhysicalMaterial
              ref={(m) => {
                mats.current[i] = m as THREE.MeshPhysicalMaterial;
              }}
              color="#7fe9ff"
              emissive="#1d6cff"
              transmission={i === 0 ? 0.92 : 0.6}
              thickness={0.6}
              roughness={0.12}
              ior={1.35}
              clearcoat={1}
              clearcoatRoughness={0.15}
              transparent
              opacity={i === 0 ? 0.6 : 0.9}
              iridescence={0.25}
            />
          )}
        </mesh>
      ))}
      <mesh geometry={halo}>
        <meshBasicMaterial ref={haloMat} color="#1d6cff" transparent opacity={0.08} side={THREE.BackSide} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      <points ref={particles}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial size={lowPower ? 0.03 : 0.022} color="#bff6ff" transparent opacity={0.7} depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
      <mesh ref={pulse} visible={false}>
        <sphereGeometry args={[0.07, 16, 16]} />
        <meshBasicMaterial color="#e8fdff" transparent opacity={0.95} blending={THREE.AdditiveBlending} />
      </mesh>
    </group>
  );
}

export default function InfinityCore3D(props: CoreProps) {
  const [visible, setVisible] = useState(!document.hidden);
  useEffect(() => {
    const onVis = () => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);
  // Reduced motion: render on demand only (a still image); hidden tab: no frames at all.
  const frameloop = !visible ? "never" : props.reducedMotion ? "demand" : "always";
  return (
    <Canvas
      frameloop={frameloop}
      dpr={props.lowPower ? [1, 1.25] : [1, 2]}
      camera={{ position: [0, 0, 4.3], fov: 42 }}
      gl={{ antialias: !props.lowPower, alpha: true, powerPreference: props.lowPower ? "low-power" : "high-performance" }}
      aria-hidden
    >
      <ambientLight intensity={0.35} />
      <pointLight position={[3, 2, 4]} intensity={40} color="#5ee7ff" />
      <pointLight position={[-3, -2, 3]} intensity={25} color="#8b7cf6" />
      <directionalLight position={[0, 4, 2]} intensity={0.6} color="#cfe9ff" />
      <Hologram {...props} />
    </Canvas>
  );
}
