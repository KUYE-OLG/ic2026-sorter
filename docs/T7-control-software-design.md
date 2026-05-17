# T7：智能分拣控制软件与 AI 模型部署说明

## 1. 主控程序架构

首版采用“同步闭环 + 可替换接口”的架构，核心由 `SorterOrchestrator` 串联相机、AI、运动、显示四个适配器；`SorterStateMachine` 负责状态约束和 watchdog。

```text
IDLE
  └─ item detected -> CLASSIFYING
CLASSIFYING
  ├─ confidence ok -> PLACING
  └─ low confidence/error -> ERROR
PLACING
  ├─ MCU ACK OK -> DISPLAYING
  └─ timeout/NACK -> ERROR
DISPLAYING
  └─ display done -> IDLE
```

当前实现路径：
- `src/sorter_control/orchestrator.py`：单件分拣主流程。
- `src/sorter_control/state_machine.py`：状态机、统计、15 s 超时保护。
- `src/sorter_control/config.py`：相机、AI、串口、显示、盒号映射配置。

## 2. 分拣控制逻辑

`process_once()` 完成一件物品闭环：

1. `camera.read()` 获取俯视图像。
2. `vision.infer()` 输出 `VisionObservation`：shape/color/text/qr/defect/confidence/bbox。
3. `DecisionFusion.decide()` 按 QR/text 优先、形状颜色兜底的规则映射到 `box_id`。
4. `motion.move_to_box()` 发送 MCU 串口命令并等待 ACK。
5. `display.show_result()` 输出序号、名称、盒号、置信度、累计成功数、图片路径。
6. 状态机统计 `total_success` 与 `per_box`。

## 3. AI 模型部署

`src/sorter_control/vision/engine.py` 预留 ONNX/TensorRT 适配入口：

- `backend=auto`：有 ONNX 文件且安装 onnxruntime 时加载 ONNX；否则进入 simulator。
- `backend=onnx`：用于笔记本或 x86 工控机验证。
- Jetson/TensorRT：后续将 ONNX 转 TensorRT engine，并在该 adapter 中补齐 YOLO postprocess。

需要现场标定后补齐的部分：
- YOLO 导出头的 bbox/class 解码。
- NMS 阈值与类别表。
- OCR/二维码 ROI 裁剪与解码。
- 像素坐标到托盘毫米坐标的 homography 标定。

## 4. 摄像头采集与预处理

`CameraCapture` 使用 OpenCV 打开固定俯视相机，配置分辨率和曝光。当前版本只做采集封装，预处理建议后续加到 `vision/engine.py`：

- 白平衡/曝光锁定。
- ROI 裁剪，剔除机械结构固定遮挡区域。
- resize 到模型输入尺寸。
- HSV 颜色校准，输出辅助 `color`。

## 5. 通信模块

`src/sorter_control/comm/protocol.py` 定义简单串口帧：

```text
命令：$MOVE,<seq>,<box_id>*<checksum>\n
事件：$ACK,<seq>,OK*<checksum>\n
```

`SerialMotionController` 发送动作命令并按 `ack_timeout_s` 等待 MCU 回应。若 MCU 返回 NACK 或超时，控制层抛出异常并由上层进入错误处理。

## 6. 显示驱动

`ConsoleDisplay` 是零依赖显示实现，适合 CLI/SSH 调试；接口为 `show_result(result, total_success)`。后续可替换为 pygame/Qt/浏览器 kiosk：

- 显示序号 `seq`。
- 显示货物名称 `item_name`。
- 显示目标盒号 `box_id`。
- 显示识别置信度和成功总数。
- 显示抓拍图片或预渲染素材路径。

## 7. 异常处理与 debug 接口

已实现：
- 状态机非法转移保护。
- 低置信度进入 ERROR。
- 15 s watchdog 超时进入 ERROR。
- 串口 checksum 校验和 ACK 超时。
- fake 组件支持无硬件端到端测试。

建议后续增加：
- JSONL 运行日志：每件物品记录图像路径、识别结果、动作 ACK、耗时。
- debug HTTP/WebSocket：实时查看状态机、最近错误、统计数。
- 失败样本自动归档到 `runs/failures/`，用于再训练。
- MCU emergency stop 与上位机 reset_error 命令。

## 8. 测试结果

本任务新增零依赖测试运行器 `run_tests.py`，当前覆盖：

- 状态机正常闭环、低置信度、超时。
- QR 优先与形状颜色兜底的决策融合。
- 串口帧编码、ACK 解析、checksum 拒绝。
- fake 组件端到端单件分拣。

验证命令：

```bash
python3 run_tests.py
python3 -m compileall -q src run_tests.py
```
