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
  const [status, setStatus] = useState("დააჭირე ▶ დაწყებას, მიმართე ცენტრი ობიექტს და დააჭირე 🎯 მონიშვნას");

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
        const maxJump = Math.max(sizeBasis * 3, Math.max(data.image_width, data.image_height) * 0.15);

        let best: Detection | null = null;
        let bestScore = Infinity;
        for (const d of (data.detections || []) as Detection[]) {
          const cx = (d.x1 + d.x2) / 2;
          const cy = (d.y1 + d.y2) / 2;
          const dist = Math.hypot(cx - selectedCentroid.x, cy - selectedCentroid.y);
          if (dist > maxJump) continue;

          const sameClass = tracked && d.class_name === tracked.class_name;

          // different-class candidates only count if they're almost exactly
          // where we last saw the object — otherwise it's probably a
          // different, unrelated object and we'd rather mark "lost"
          if (!sameClass) {
            const tightRadius = Math.max(sizeBasis * 0.6, 20);
            if (dist > tightRadius) continue;
          }

          let sizePenalty = 0;
          if (selectedSize) {
            const dw = d.x2 - d.x1, dh = d.y2 - d.y1;
            const ratio = Math.max(dw / selectedSize.w, selectedSize.w / dw,
                                    dh / selectedSize.h, selectedSize.h / dh);
            if (ratio > 2.2) sizePenalty = maxJump * 0.6;
          }

          const confBonus = (1 - d.confidence) * maxJump * 0.15;
          const classPenalty = sameClass ? 0 : maxJump * 0.3;

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

  function selectAtCenter() {
    if (!detections.length) {
      setStatus("⚠️ ჯერ არაფერია აღმოჩენილი — მიმართე კამერა ობიექტს");
      return;
    }
    const centerX = imgSize.w / 2;
    const centerY = imgSize.h / 2;

    let best: Detection | null = null;
    let bestDist = Infinity;
    for (const d of detections) {
      const cx = (d.x1 + d.x2) / 2;
      const cy = (d.y1 + d.y2) / 2;
      const inside = centerX >= d.x1 && centerX <= d.x2 && centerY >= d.y1 && centerY <= d.y2;
      const dist = Math.hypot(cx - centerX, cy - centerY);
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
        <View style={StyleSheet.absoluteFill} pointerEvents="none">
          <Svg width={SCREEN_W} height={STAGE_H}>
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
          {/* center crosshair — line up the target here, then tap "მონიშვნა" */}
          <View style={styles.crosshair}>
            <View style={styles.crosshairH} />
            <View style={styles.crosshairV} />
            <View style={styles.crosshairDot} />
          </View>
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
        <TouchableOpacity style={styles.btn} onPress={selectAtCenter}>
          <Text style={styles.btnText}>🎯 მონიშვნა</Text>
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
          <Text style={styles.btnText}>✖ მოხსნა</Text>
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
  crosshair: {
    position: "absolute", top: "50%", left: "50%",
    width: 34, height: 34, marginLeft: -17, marginTop: -17,
  },
  crosshairH: {
    position: "absolute", top: 16, left: 2, right: 2, height: 2,
    backgroundColor: "rgba(255,255,255,0.85)",
  },
  crosshairV: {
    position: "absolute", left: 16, top: 2, bottom: 2, width: 2,
    backgroundColor: "rgba(255,255,255,0.85)",
  },
  crosshairDot: {
    position: "absolute", top: 14, left: 14, width: 6, height: 6,
    borderRadius: 3, backgroundColor: "#4dd0ff",
  },
});
