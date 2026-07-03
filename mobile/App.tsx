/**
 * BLACK FALCON — Mobile Tracking Demo
 * ------------------------------------
 * Same idea as the web demo: capture frames from the phone camera,
 * send them to the backend /detect endpoint, draw boxes for every
 * detected object, and keep tracking whichever one the user tapped.
 *
 * This is a minimal starting scaffold — wire it into your existing
 * NEXUS-style Expo project structure as needed.
 */

import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Dimensions,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import Svg, { Rect, Text as SvgText } from "react-native-svg";

const { width: SCREEN_W } = Dimensions.get("window");
const STAGE_H = (SCREEN_W * 3) / 4;

// Backend API URL — hardcoded so the app works without any setup.
// Change this if you redeploy the backend under a different address.
const API_URL = "https://black-falcon-track.onrender.com";

type Detection = {
  class_name: string;
  confidence: number;
  x1: number; y1: number; x2: number; y2: number;
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const [running, setRunning] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 });
  const [selectedCentroid, setSelectedCentroid] = useState<{ x: number; y: number } | null>(null);
  const [selectedSize, setSelectedSize] = useState<{ w: number; h: number } | null>(null);
  const [tracked, setTracked] = useState<Detection | null>(null);
  const [isLost, setIsLost] = useState(false);
  const [status, setStatus] = useState("ჩაწერე backend URL და დააჭირე Start-ს");

  const captureLoop = useCallback(async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.5,
        skipProcessing: true,
      });
      if (!photo?.base64) return;

      const form = new FormData();
      form.append("file", {
        uri: photo.uri,
        name: "frame.jpg",
        type: "image/jpeg",
      } as any);

      const base = API_URL.replace(/\/$/, "");
      const res = await fetch(`${base}/detect`, { method: "POST", body: form });
      const data = await res.json();

      setDetections(data.detections || []);
      setImgSize({ w: data.image_width, h: data.image_height });

      if (selectedCentroid) {
        const sizeBasis = selectedSize ? Math.max(selectedSize.w, selectedSize.h) : 0;
        const maxJump = Math.max(sizeBasis * 3, Math.max(data.image_width, data.image_height) * 0.2);

        let best: Detection | null = null;
        let bestScore = Infinity;
        for (const d of (data.detections || []) as Detection[]) {
          const cx = (d.x1 + d.x2) / 2;
          const cy = (d.y1 + d.y2) / 2;
          const dist = Math.hypot(cx - selectedCentroid.x, cy - selectedCentroid.y);
          if (dist > maxJump) continue;

          const classPenalty = (tracked && d.class_name === tracked.class_name) ? 0 : maxJump * 0.3;

          let sizePenalty = 0;
          if (selectedSize) {
            const dw = d.x2 - d.x1, dh = d.y2 - d.y1;
            const ratio = Math.max(dw / selectedSize.w, selectedSize.w / dw,
                                    dh / selectedSize.h, selectedSize.h / dh);
            if (ratio > 2.2) sizePenalty = maxJump * 0.6;
          }

          const confBonus = (1 - d.confidence) * maxJump * 0.15;

          const score = dist + classPenalty + sizePenalty + confBonus;
          if (score < bestScore) { bestScore = score; best = d; }
        }

        if (best) {
          // 🟢 FOUND
          setTracked(best);
          setSelectedCentroid({ x: (best.x1 + best.x2) / 2, y: (best.y1 + best.y2) / 2 });
          setSelectedSize({ w: best.x2 - best.x1, h: best.y2 - best.y1 });
          setIsLost(false);
          setStatus(`🟢 ვხედავთ: ${best.class_name}`);
        } else {
          // 🔴 LOST — keep last known box, don't move it
          setIsLost(true);
          setStatus("🔴 ობიექტი დაიკარგა — ვეძებ...");
        }
      }
    } catch (e) {
      setStatus("⚠️ backend-თან კავშირის შეცდომა");
    }
  }, [selectedCentroid, selectedSize, tracked]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(captureLoop, 900);
    return () => clearInterval(id);
  }, [running, captureLoop]);

  if (!permission) return <View style={styles.center}><Text>...</Text></View>;
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>საჭიროა კამერაზე წვდომა</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>ნებართვის მიცემა</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const sx = SCREEN_W / imgSize.w;
  const sy = STAGE_H / imgSize.h;

  function handleTap(evt: any) {
    if (!detections.length) return;
    const { locationX, locationY } = evt.nativeEvent;
    const clickX = locationX / sx;
    const clickY = locationY / sy;

    let best: Detection | null = null;
    let bestDist = Infinity;
    for (const d of detections) {
      const cx = (d.x1 + d.x2) / 2;
      const cy = (d.y1 + d.y2) / 2;
      const inside = clickX >= d.x1 && clickX <= d.x2 && clickY >= d.y1 && clickY <= d.y2;
      const dist = Math.hypot(cx - clickX, cy - clickY);
      if (inside) { best = d; bestDist = -1; break; }
      if (dist < bestDist) { bestDist = dist; best = d; }
    }
    if (best) {
      setTracked(best);
      setSelectedCentroid({ x: (best.x1 + best.x2) / 2, y: (best.y1 + best.y2) / 2 });
      setSelectedSize({ w: best.x2 - best.x1, h: best.y2 - best.y1 });
      setIsLost(false);
      setStatus(`🟢 მონიშნულია: ${best.class_name}`);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>BLACK <Text style={{ color: "#ff5c5c" }}>FALCON</Text></Text>

      <View style={{ width: SCREEN_W, height: STAGE_H }}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        <View style={StyleSheet.absoluteFill} onTouchEnd={handleTap}>
          <Svg width={SCREEN_W} height={STAGE_H}>
            {detections.map((d, i) => (
              <React.Fragment key={i}>
                <Rect
                  x={d.x1 * sx} y={d.y1 * sy}
                  width={(d.x2 - d.x1) * sx} height={(d.y2 - d.y1) * sy}
                  stroke="#2ecc71" strokeWidth={2} fill="none"
                />
              </React.Fragment>
            ))}
            {tracked && (
              <Rect
                x={tracked.x1 * sx} y={tracked.y1 * sy}
                width={(tracked.x2 - tracked.x1) * sx} height={(tracked.y2 - tracked.y1) * sy}
                stroke={isLost ? "#ff5c5c" : "#2ecc71"}
                strokeDasharray={isLost ? "8,6" : undefined}
                strokeWidth={3} fill="none"
              />
            )}
          </Svg>
        </View>
      </View>

      <Text style={styles.status}>{status}</Text>

      <View style={styles.row}>
        <TouchableOpacity
          style={[styles.btn, running && styles.btnActive]}
          onPress={() => setRunning(r => !r)}
        >
          <Text style={styles.btnText}>{running ? "⏸ გაჩერება" : "▶ დაწყება"}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.btn}
          onPress={() => {
            setTracked(null);
            setSelectedCentroid(null);
            setSelectedSize(null);
            setIsLost(false);
          }}
        >
          <Text style={styles.btnText}>✖ მონიშვნის მოხსნა</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0e14", alignItems: "center", paddingTop: 50 },
  center: { flex: 1, backgroundColor: "#0a0e14", justifyContent: "center", alignItems: "center" },
  title: { color: "#e6edf3", fontSize: 20, fontWeight: "700", letterSpacing: 2, marginBottom: 10 },
  text: { color: "#e6edf3", marginBottom: 12 },
  input: {
    width: SCREEN_W - 32, backgroundColor: "#121821", color: "#e6edf3",
    borderRadius: 8, borderWidth: 1, borderColor: "#232b36",
    paddingHorizontal: 12, paddingVertical: 8, marginBottom: 12,
  },
  status: { color: "#7d8590", fontSize: 12, marginTop: 10, textAlign: "center", paddingHorizontal: 20 },
  row: { flexDirection: "row", gap: 10, marginTop: 14 },
  btn: {
    backgroundColor: "#121821", borderWidth: 1, borderColor: "#232b36",
    paddingHorizontal: 16, paddingVertical: 10, borderRadius: 8,
  },
  btnActive: { backgroundColor: "#ff5c5c", borderColor: "#ff5c5c" },
  btnText: { color: "#e6edf3", fontWeight: "600" },
});
