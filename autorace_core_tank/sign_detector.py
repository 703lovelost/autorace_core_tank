from typing import Optional, Dict, Any
import numpy as np
import cv2


def find_sign_roi(
    bgr: np.ndarray,
    *,
    min_area: float = 2000.0,
    circ_min: float = 0.6,
    circ_max: float = 1.2,
    demonstration: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    if bgr is None:
        return None

    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return None

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < float(min_area):
        return None

    perimeter = float(cv2.arcLength(largest, True))
    if perimeter <= 0.0:
        return None

    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    if not (float(circ_min) <= circularity <= float(circ_max)):
        return None

    x, y, w, h = cv2.boundingRect(largest)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(bgr.shape[1], x + w)
    y1 = min(bgr.shape[0], y + h)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    center = (int(x0 + w // 2), int(y0 + h // 2))

    try:
        cv2.imshow("sign", roi)
    except Exception:
        pass

    if demonstration is not None:
        try:
            cv2.line(demonstration, (center[0], 0), (center[0], demonstration.shape[0]), [255, 0, 0], 5)
        except Exception:
            pass

    return {
        "bbox": (int(x0), int(y0), int(w), int(h)),
        "center": center,
        "area": area,
        "circularity": circularity,
        "roi": roi,
    }
