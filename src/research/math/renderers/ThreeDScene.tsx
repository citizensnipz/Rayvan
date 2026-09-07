import {
  Component,
  Suspense,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type ComponentRef,
} from "react";
import { Canvas, useThree } from "@react-three/fiber";
import {
  Html,
  Line,
  OrbitControls,
  TransformControls,
} from "@react-three/drei";
import {
  DoubleSide,
  Quaternion,
  Vector3,
  type Mesh,
  type BufferGeometry,
} from "three";
import { length, minus, plus, times, type V3 } from "../core/geometry";
import type {
  GeometryScene,
  SceneArrow,
  ScenePoint,
  SceneSurface,
} from "./scene";

function Label({ position, children }: { position: V3; children: ReactNode }) {
  return (
    <Html position={[...position]} center style={{ pointerEvents: "none" }}>
      <span className="math-3d-label">{children}</span>
    </Html>
  );
}
export function VectorArrow3D({
  from = [0, 0, 0],
  to,
  color = "#d8ff75",
  label,
}: SceneArrow) {
  const delta = minus(to, from),
    size = length(delta),
    middle = plus(from, times(delta, 0.5));
  const quaternion = useMemo(
    () =>
      new Quaternion().setFromUnitVectors(
        new Vector3(0, 1, 0),
        new Vector3(...delta).normalize(),
      ),
    [...delta],
  );
  return (
    <group>
      {size > 1e-8 && (
        <>
          <mesh position={[...middle]} quaternion={quaternion}>
            <cylinderGeometry args={[0.016, 0.016, size, 8]} />
            <meshBasicMaterial color={color} />
          </mesh>
          <mesh position={[...to]} quaternion={quaternion}>
            <coneGeometry args={[0.065, Math.min(0.2, size * 0.3), 10]} />
            <meshBasicMaterial color={color} />
          </mesh>
        </>
      )}
      {label && <Label position={plus(to, [0.1, 0.1, 0.1])}>{label}</Label>}
    </group>
  );
}
export function CoordinateAxes3D() {
  return (
    <group>
      {(
        [
          [3, 0, 0],
          [0, 3, 0],
          [0, 0, 3],
        ] as V3[]
      ).map((to, i) => (
        <VectorArrow3D
          key={i}
          to={to}
          color={["#ef7b86", "#d8ff75", "#38c6cc"][i]}
          label={["x", "y", "z"][i]}
        />
      ))}
    </group>
  );
}
export function GridPlane3D() {
  return (
    <gridHelper
      args={[10, 20, "#435063", "#263142"]}
      rotation={[Math.PI / 2, 0, 0]}
    />
  );
}
export function PointCloud3D({
  points,
  color,
  size = 0.055,
}: {
  points: readonly V3[];
  color: string;
  size?: number;
}) {
  const positions = useMemo(() => new Float32Array(points.flat()), [points]);
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color={color} size={size} sizeAttenuation />
    </points>
  );
}
export function SurfaceMesh3D({
  vertices,
  indices,
  color = "#38c6cc",
  opacity = 0.5,
}: SceneSurface) {
  const ref = useRef<BufferGeometry>(null);
  const { invalidate } = useThree();
  const positions = useMemo(
    () => new Float32Array(vertices.flat()),
    [vertices],
  );
  const index = useMemo(() => new Uint16Array(indices), [indices]);
  useLayoutEffect(() => {
    ref.current?.computeVertexNormals();
    invalidate();
  }, [positions, index, invalidate]);
  return (
    <mesh>
      <bufferGeometry ref={ref}>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="index" args={[index, 1]} />
      </bufferGeometry>
      <meshStandardMaterial
        color={color}
        side={DoubleSide}
        transparent
        opacity={opacity}
        roughness={0.85}
        depthWrite={false}
      />
    </mesh>
  );
}
export function Trajectory3D({
  points,
  color = "#d8ff75",
  dashed = false,
}: {
  points: readonly V3[];
  color?: string;
  dashed?: boolean;
}) {
  return points.length > 1 ? (
    <Line
      points={points.map((p) => [...p] as [number, number, number])}
      color={color}
      lineWidth={1.7}
      dashed={dashed}
      dashSize={0.1}
      gapSize={0.06}
    />
  ) : null;
}
function DraggablePoint({
  position,
  color = "#d8ff75",
  radius = 0.085,
  label,
  onMove,
}: ScenePoint) {
  const mesh = useRef<Mesh>(null);
  const body = (
    <mesh ref={mesh} position={[...position]}>
      <sphereGeometry args={[radius, 12, 8]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
  return (
    <>
      {onMove ? (
        <TransformControls
          mode="translate"
          size={0.55}
          onObjectChange={() => {
            if (mesh.current) {
              const next = mesh.current.position
                .toArray()
                .map((n) => Math.max(-4, Math.min(4, n))) as [
                number,
                number,
                number,
              ];
              if (length(minus(next, position)) > 1e-6) onMove(next);
            }
          }}
        >
          {body}
        </TransformControls>
      ) : (
        body
      )}
      {label && <Label position={plus(position, [0, 0, 0.2])}>{label}</Label>}
    </>
  );
}
function cameraFrame(scene: GeometryScene) {
  const points: V3[] = [
    [-3, -3, -1],
    [3, 3, 3],
    ...(scene.points?.map((p) => p.position) ?? []),
    ...(scene.clouds?.flatMap((c) => c.points) ?? []),
    ...(scene.surfaces?.flatMap((s) => s.vertices) ?? []),
    ...(scene.paths?.flatMap((p) => p.points) ?? []),
    ...(scene.arrows?.flatMap((a) => [a.from ?? ([0, 0, 0] as V3), a.to]) ??
      []),
    ...(scene.regions?.flatMap((r) => [
      plus(r.center, [r.radius, r.radius, r.radius]),
      minus(r.center, [r.radius, r.radius, r.radius]),
    ]) ?? []),
  ];
  const lower = [0, 1, 2].map((i) => Math.min(...points.map((p) => p[i]))),
    upper = [0, 1, 2].map((i) => Math.max(...points.map((p) => p[i])));
  const center = lower.map((n, i) => (n + upper[i]) / 2) as [
      number,
      number,
      number,
    ],
    distance = Math.max(
      11,
      length(minus(upper as unknown as V3, lower as unknown as V3)) * 1.5,
    );
  return { center, distance };
}
export function OrbitCameraControls({
  reset,
  frame,
}: {
  reset: number;
  frame: ReturnType<typeof cameraFrame>;
}) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { camera, invalidate } = useThree();
  useEffect(() => {
    camera.up.set(0, 0, 1);
    camera.position.set(
      frame.center[0] + frame.distance * 0.5,
      frame.center[1] - frame.distance * 0.65,
      frame.center[2] + frame.distance * 0.5,
    );
    camera.lookAt(...frame.center);
    controls.current?.target.set(...frame.center);
    controls.current?.update();
    invalidate();
  }, [reset, camera, invalidate, frame]);
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping={false}
      minDistance={1}
      maxDistance={80}
    />
  );
}
class SceneBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <p role="alert" className="math-help">
        3D rendering failed. Switch to 2D or reopen this concept after checking
        WebGL2 support.
      </p>
    ) : (
      this.props.children
    );
  }
}
export default function ThreeDScene({ scene }: { scene: GeometryScene }) {
  const [reset, setReset] = useState(0),
    [lost, setLost] = useState(false);
  const [frame, setFrame] = useState(() => cameraFrame(scene));
  return (
    <div className="math-3d-wrap">
      <div className="math-3d-toolbar">
        <span>
          Drag to orbit · scroll to zoom · coloured handles move editable points
        </span>
        <button
          onClick={() => {
            setFrame(cameraFrame(scene));
            setReset((n) => n + 1);
          }}
        >
          Reset camera
        </button>
      </div>
      {lost ? (
        <p role="alert">
          WebGL context lost. Reopen this concept to restore the scene.
        </p>
      ) : (
        <SceneBoundary>
          <Canvas
            frameloop="demand"
            dpr={[1, 1.5]}
            camera={{
              position: [7, -9, 7],
              up: [0, 0, 1],
              fov: 45,
              near: 0.05,
              far: 300,
            }}
            gl={{ antialias: true, powerPreference: "low-power" }}
            onCreated={({ gl }) => {
              gl.domElement.addEventListener(
                "webglcontextlost",
                () => setLost(true),
                { once: true },
              );
            }}
          >
            <color attach="background" args={["#0f141c"]} />
            <ambientLight intensity={1.2} />
            <directionalLight position={[5, -4, 8]} intensity={2} />
            <OrbitCameraControls reset={reset} frame={frame} />
            <CoordinateAxes3D />
            <GridPlane3D />
            <Suspense fallback={null}>
              {scene.clouds?.map((cloud, i) => (
                <PointCloud3D key={i} {...cloud} />
              ))}
              {scene.surfaces?.map((surface, i) => (
                <SurfaceMesh3D key={i} {...surface} />
              ))}
              {scene.paths?.map((path, i) => (
                <Trajectory3D key={i} {...path} />
              ))}
              {scene.arrows?.map((arrow, i) => (
                <VectorArrow3D key={i} {...arrow} />
              ))}
              {scene.points?.map((point, i) => (
                <DraggablePoint key={i} {...point} />
              ))}
              {scene.regions?.map((region, i) => (
                <mesh key={i} position={[...region.center]}>
                  <sphereGeometry args={[region.radius, 20, 12]} />
                  <meshBasicMaterial
                    color={region.color}
                    wireframe
                    transparent
                    opacity={0.12}
                    depthWrite={false}
                  />
                </mesh>
              ))}
            </Suspense>
          </Canvas>
        </SceneBoundary>
      )}
    </div>
  );
}
