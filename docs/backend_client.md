# backend_client.py

## 中文说明

### 模块职责

`backend_client.py` 负责把检测结果以 JSON 形式异步发送到后端。它使用 Python 标准库实现，不额外增加树莓派部署依赖。

### 主要类

- `BackendClient`：轻量级后台 HTTP JSON 客户端。

### 主要方法

- `__init__(url, timeout=1.0, max_queue=8)`：如果提供了后端 URL，则启动后台线程；如果没有 URL，则客户端处于禁用状态。
- `submit(payload)`：提交一条 JSON payload。队列满时会丢弃旧数据，优先保留新检测结果，避免摄像头循环被慢后端拖住。
- `close()`：通知后台线程停止，并等待最多 2 秒退出。
- `_worker()`：后台线程循环，从队列取数据并发送。
- `_post_json(payload)`：把字典编码成 UTF-8 JSON，通过 HTTP POST 发送到后端。

### 后端接口

请求方法：`POST`

请求头：

```text
Content-Type: application/json; charset=utf-8
```

请求体包含设备 ID、时间戳、帧号、图像尺寸、ROI、FPS、推理耗时和检测框列表。后端返回任意 `2xx` 状态即可。

### 错误处理

后端不可用、超时或网络错误不会中断检测主循环。错误日志最多约每 5 秒打印一次，避免刷屏。

## English Notes

### Module Purpose

`backend_client.py` asynchronously posts detection JSON payloads to a backend. It uses only the Python standard library to keep Raspberry Pi deployment simple.

### Key Class

- `BackendClient`: Small background HTTP JSON sender.

### Key Methods

- `__init__(url, timeout=1.0, max_queue=8)`: Starts a background worker when a URL is provided.
- `submit(payload)`: Queues one payload. If the queue is full, older payloads are dropped so the detector loop stays responsive.
- `close()`: Stops the worker thread.
- `_worker()`: Pulls payloads from the queue and posts them.
- `_post_json(payload)`: Encodes and sends one JSON HTTP POST request.

### Backend Contract

The backend should accept `POST` requests with `Content-Type: application/json; charset=utf-8`. Any `2xx` response is enough; the client does not require a response body.

### Failure Handling

Timeouts and network failures do not stop inference. Errors are rate-limited to avoid noisy logs.
