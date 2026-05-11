# Backend Requirements

## 中文说明

### 部署角色

- 树莓派视频端：运行 `mjpg-streamer`，只负责采集摄像头并输出原始 MJPEG 视频流。
- PC 检测端：读取树莓派 MJPEG 网络流，运行 NCNN 推理，POST JSON 检测结果。
- PC 后端：接收 JSON、拉取或代理树莓派 MJPEG、缓存最新检测状态、在前端页面叠加渲染检测框。

### 检测端输出

原始视频流由树莓派 `mjpg-streamer` 提供：

```text
GET http://172.20.10.2:8080/?action=stream
Content-Type: multipart/x-mixed-replace; boundary=frame
```

检测 JSON 由 PC 检测端提供：

```text
POST http://127.0.0.1:8000/api/detections
Content-Type: application/json; charset=utf-8
```

视频帧不放进 JSON。JSON 只传检测元数据和检测框。

### 设备配置

PC 后端维护设备配置表：

```text
pi-camera-01 -> http://172.20.10.2:8080/?action=stream
win-camera-01 -> http://127.0.0.1:8090/stream.mjpg
```

检测端 JSON 不负责自动注册 `stream_url`。后端按 `device_id` 查配置表。

### JSON Payload

```json
{
  "device_id": "pi-camera-01",
  "timestamp_ms": 1777890000000,
  "frame_id": 128,
  "image_size": {"width": 640, "height": 640},
  "roi": {"x1": 0, "y1": 0, "x2": 320, "y2": 640, "width": 320, "height": 640, "reason": "tile"},
  "fps": 18.42,
  "inference_ms": 12.37,
  "detections": [
    {
      "class_id": 0,
      "label": "Correct Wear",
      "confidence": 0.9345,
      "cached": false,
      "box": {"x1": 160, "y1": 96, "x2": 236, "y2": 210}
    }
  ]
}
```

后端返回任意 `2xx` 即可，响应体可为空。建议后端保存 `received_at_ms`，用于判断检测数据是否延迟。

### 违规者追踪与统计规则

检测模型不区分不同人脸，检测端也不需要输出人员 ID。PC 后端按 `device_id` 为每路视频维护独立的追踪状态，并为每个稳定目标生成内部 `track_id`。每日违规人数统计按“人”结算，而不是按帧或检测框累计。

#### 规则一：双重空间匹配

后端匹配新旧检测框时同时使用 IoU 重合度和中心点欧氏距离，避免人员靠近镜头时因框体快速变大导致 IoU 暴跌而被误判为新目标。

- 每帧收到 `detections[]` 后，先计算每个检测框与现有存活 `track_id` 的 IoU。
- 再计算检测框中心点与历史框中心点的欧氏距离，距离建议按画面宽高归一化，避免不同分辨率阈值不一致。
- 满足 `iou >= iou_threshold` 或 `center_distance <= center_distance_threshold` 的检测框，可匹配到同一个 `track_id`。
- 当多个目标同时满足条件时，优先选择综合代价最低的目标，建议代价为 `1 - iou + normalized_center_distance`。
- 默认建议：`iou_threshold = 0.3`，`center_distance_threshold = 0.08`（归一化到画面对角线），实际值可按现场摄像头视角调参。

#### 规则二：状态防抖队列

后端为每个 `track_id` 建立固定长度滑动窗口，用多数表决得到稳定状态，避免低头、转脸、遮挡等造成“合规/违规”状态闪烁。

- 每个 `track_id` 保存最近 `N` 帧的状态队列，默认 `N = 10`。
- 检测框匹配到某个 `track_id` 后，将当前帧的状态压入队列。
- `label` 或 `class_id` 需要映射为二值状态：`violation` 或 `compliant`。
- 当队列中违规比例 `> violation_ratio_threshold` 时，当前稳定状态判定为违规。
- 默认建议：`violation_ratio_threshold = 0.7`。队列未填满时也可按已有样本计算，但样本数过少时不建议立即结算。
- 一旦某个 `track_id` 的稳定状态达到违规，应记录 `ever_violated = true`，用于离开画面后的最终人数结算。

#### 规则三：生命周期管理

每个 `track_id` 需要维护完整生命周期：诞生 -> 追踪 -> 隐身缓冲 -> 死亡结算。后端不能因为某几帧未收到检测框就立刻销毁目标。

- `birth`：新检测框无法匹配任何现有 `track_id` 时，创建新 `track_id`。
- `tracking`：检测框持续匹配成功时，更新最后检测框、最后出现时间、状态防抖队列。
- `lost`：某个 `track_id` 在当前帧未匹配到检测框时，进入或保持隐身缓冲状态。
- `death`：若 `now_ms - last_seen_ms > patience_ms`，判定该目标已离开画面，执行一次结算后删除或归档。
- 默认建议：`patience_ms = 3000`。3 秒内目标重新出现并被空间匹配命中时，继续使用原 `track_id`，不新建档案。
- 死亡结算时，如果 `ever_violated = true`，则当日该设备或全局的违规人数 `+1`。同一个 `track_id` 只允许结算一次，避免重复计数。
- 即使检测端短时间没有 POST 新 JSON，后端也应通过定时任务或请求时懒清理检查超时 `track_id`，保证离开画面的目标最终会结算。

建议后端为每个 `track_id` 至少维护以下字段：

```json
{
  "track_id": "pi-camera-01-000001",
  "device_id": "pi-camera-01",
  "created_at_ms": 1777890000000,
  "last_seen_ms": 1777890001200,
  "last_box": {"x1": 160, "y1": 96, "x2": 236, "y2": 210},
  "state_window": ["violation", "compliant", "violation"],
  "stable_state": "violation",
  "ever_violated": true,
  "settled": false
}
```

统计口径：

- `current_violators`：当前仍在画面内且稳定状态为违规的 `track_id` 数量。
- `today_violation_count`：今日已死亡结算且 `ever_violated = true` 的人数总数。
- `active_tracks`：当前处于 `tracking` 或 `lost` 且未超过 `patience_ms` 的目标列表。
- 统计应按自然日重置；若系统跨时区部署，需要明确以后端本地时区或配置时区为准。

### 后端接口建议

- `POST /api/detections`：接收检测端 JSON，按 `device_id` 覆盖保存最新状态。
- `GET /devices/{device_id}/stream.mjpg`：代理设备 MJPEG，前端只访问 PC 后端。
- `GET /devices/{device_id}/latest`：返回设备最新 JSON 和后端接收时间。
- `GET /devices/{device_id}/tracks`：返回该设备当前 `active_tracks`、稳定状态和追踪框。
- `GET /stats/today`：返回今日违规人数、当前违规人数、活跃追踪人数等统计。
- `GET /devices/{device_id}/preview`：展示预览页面。

### 前端渲染规则

- 视频层播放后端代理的 MJPEG 原始流。
- 覆盖层使用最新 JSON 的 `detections[].box` 画框。
- 坐标按 `image_size` 等比缩放到页面视频显示区域。
- `cached=true` 建议画成虚线或浅色，表示该框来自检测缓存。
- 超过 3 秒未收到 JSON：显示“检测数据延迟”。
- MJPEG 拉流失败：显示“视频流断开”。
- 视频断开和 JSON 延迟独立处理。

## English Notes

The recommended local-test path is Raspberry Pi `mjpg-streamer` for raw MJPEG, PC-side NCNN detection for JSON posting, and a PC backend for video proxying, latest-state caching, and browser rendering. The detector's built-in `--raw-stream` remains a fallback when no external streamer is available.

The backend should keep a `device_id -> stream_url` table, accept `POST /api/detections`, and render the latest JSON over the latest MJPEG frame. Frame-level synchronization is not required; use the latest available detection state.
