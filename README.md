# IC2026 智能分拣控制软件

本目录实现 T7 控制软件首版骨架：状态机、多模块调度、识别决策融合、运动控制通信协议、显示输出、ONNX/TensorRT 部署适配入口和可离线测试的 fake 组件。

## 架构

```text
CameraCapture -> VisionEngine -> DecisionFusion -> SorterStateMachine
                                      |                    |
                                      v                    v
                              SerialMotionController -> ConsoleDisplay
```

关键约束：
- 每件物品按 `采集 -> 识别 -> 分类映射 -> 执行动作 -> 显示统计` 闭环处理。
- 状态机内置 15 s watchdog，低置信度/超时/非法状态会进入 ERROR。
- `configs/task_config.yaml` 维护现场可改的储物盒映射，避免把任务规则写死到模型或代码里。
- `VisionEngine` 优先加载 ONNX，未安装 onnxruntime 或模型缺失时进入 simulator，便于笔记本端调试。

## 本地验证

当前环境没有 `python3-venv`/pytest，因此提供零依赖测试运行器：

```bash
python3 run_tests.py
```

完整开发环境建议：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 板端部署建议

Jetson/TensorRT 路线：
1. 训练端导出 `models/sorter_yolo.onnx`。
2. 在板端使用 TensorRT 或 onnxruntime-gpu 构建 engine。
3. 在 `src/sorter_control/vision/engine.py` 中补齐与实际 YOLO 导出头匹配的 postprocess：NMS、类别映射、ROI 裁剪、OCR/QR 解码。
4. 使用 `configs/task_config.yaml` 调整盒号映射和串口名。

## 串口协议

命令帧：`$MOVE,<seq>,<box_id>*<checksum>\n`

事件帧：`$ACK,<seq>,OK*<checksum>\n`

checksum 为带 salt 的 XOR，用于快速发现截断/串扰；正式 MCU 固件需使用同一算法。
