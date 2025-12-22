# autorace_core_tank

This package contains a simple RGB+Depth lane PID controller and a local traffic sign classifier.

## What was added

- `models/` folder with a local copy of the ShuffleNet (GTSRB, 43 classes) weights from the repository `YusufBayram-Personal/shufflenet_GTSRB_multi_krum_round_11`.
- New node `autorace_core_tank/sign_detector_node.py` (console script: `sign_detector`). It subscribes to `/color/image`, runs ROI extraction for blue circular signs + ShuffleNet classification, publishes:
  - `/autorace_core_tank/sign_detections` (`std_msgs/String` with JSON payload)
  - `/autorace_core_tank/sign_debug_image` (`sensor_msgs/Image`)
- `lane_pid_node.py` now loads the same local model and uses it inside `look_for_sign()`:
  - sets `sign_found=True` when a sign is detected and classified above `sign.conf_threshold`
  - when the sign is classified as **Turn left ahead** or **Turn right ahead**, the robot performs a sharp turn and then returns to `state=2`
- Launch and config were updated to start both nodes.

## Running

```bash
ros2 launch autorace_core_tank autorace_core.launch.py
```

The model is loaded only from the local package share directory (`share/autorace_core_tank/models/pytorch_model.bin`).
