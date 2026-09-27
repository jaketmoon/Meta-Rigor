# MetaRigor

MetaRigor 是面向 Agent 的证据审查服务。仓库通过一个可安装的 CLI 暴露稳定的 JSON 接口，外部 Agent 不需要了解内部 Python 模块即可调用。

## 安装

要求 Python 3.12 或更高版本：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

如果需要运行 JSON Schema 校验，请安装完整依赖：

```bash
.venv/bin/pip install -r requirements.txt
```

安装后可使用：

```bash
metarigor --help
```

也可以不安装，直接运行：

```bash
PYTHONPATH=backend/src python3 -m metarigor --help
```

## CLI

检查服务：

```bash
metarigor health
```

查看 Agent 能力和协议版本：

```bash
metarigor capabilities
```

当前提供的基础操作：

- `health`：健康检查
- `capabilities`：返回能力清单
- `hash`：计算字符串 SHA-256
- `validate`：使用 JSON Schema 校验输入

例如计算哈希：

```bash
echo '{"value":"hello"}' | metarigor run hash
```

例如校验 JSON：

```bash
cat <<'JSON' | metarigor run validate
{
  "schema": {"type": "object", "required": ["case_id"]},
  "value": {"case_id": "demo"}
}
JSON
```

## Agent 接入

`metarigor agent` 使用 JSONL（每行一个 JSON 对象）作为 stdin/stdout 协议，适合被其他 Agent、工作流编排器或沙箱进程调用。

启动服务：

```bash
metarigor agent
```

发送请求：

```json
{"id":"req-1","operation":"health","input":{}}
{"id":"req-2","operation":"hash","input":{"value":"hello"}}
```

每个请求都会返回一个 JSON 对象：

```json
{
  "protocol": "metarigor-agent-v1",
  "id": "req-2",
  "ok": true,
  "result": {
    "algorithm": "sha256",
    "digest": "..."
  }
}
```

错误不会混入日志格式，而是以结构化响应返回：

```json
{
  "protocol": "metarigor-agent-v1",
  "id": "req-3",
  "ok": false,
  "error": {
    "type": "ValueError",
    "message": "..."
  }
}
```

单次调用可以使用：

```bash
echo '{"id":"one-shot","operation":"health","input":{}}' \
  | metarigor agent --once
```

## 项目结构

```text
backend/src/metarigor/
├── application/       # 产品服务边界和操作分发
├── cli/                # 命令行入口与 JSONL Agent 适配器
├── adapters/           # 外部模型和运行时适配器
├── data_extraction/    # 数据提取能力
├── evidence_acquisition/  # 证据获取能力
├── evidence_certainty/    # 证据确定性评估
├── manuscript/         # 稿件相关能力
└── risk_of_bias_v3/    # 偏倚风险评估
```

论文实验脚本和冻结数据仍保留在仓库中，但产品集成应优先使用 `metarigor` CLI 和 `metarigor-agent-v1` 协议。

## 开发检查

```bash
PYTHONPATH=backend/src python3 -m compileall -q backend/src
PYTHONPATH=backend/src python3 -m metarigor health
```


