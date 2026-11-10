# 劳动法智能问答助手（RAG）

课程作业级 RAG 小项目：用户提问劳动法问题 → 系统在本地法规库中做「向量 + 关键词」混合检索 → 只把命中的条文送进大模型 → 模型仅依据条文作答，并强制标注《法规全名》第X条；依据不足时第一句直接回答"根据现有法条无法确定"。

## 目录说明

```
LaborLawRAG/
├── data/
│   ├── raw/          # 原始法规 txt（一行条文一段），已内置劳动合同法、劳动法、年休假条例等 5 部
│   ├── raw_clean/    # clean.py 清洗后的文本
│   └── chunks.jsonl  # 切块产物，每行一个 JSON（id/law/article_no/chapter/text/part）
├── index/
│   ├── chroma/       # 向量库持久化目录（chromadb PersistentClient）
│   └── bm25.pkl      # (chunk_ids, BM25Okapi)
├── src/
│   ├── config.py     # 全部可调参数与 .env 读取
│   ├── clean.py      # 文本清洗（页码/页眉页脚/空行/全角标点）
│   ├── chunker.py    # 按"第X条"切分，超长条文按款二次切分
│   ├── build_index.py# 建向量库 + BM25
│   ├── retriever.py  # 向量 / BM25 / RRF 混合检索
│   ├── prompt.py     # 规则化 Prompt 模板
│   ├── generator.py  # 调用 OpenAI 兼容接口（重试 1 次）
│   ├── citation.py   # 引用溯源校验
│   └── evaluate.py   # 4 组配置离线评测
├── eval/
│   ├── testset.csv   # 15 道标注题 + 15 行空模板
│   └── results/      # 明细与 metrics.csv
├── app.py            # Streamlit 界面
├── requirements.txt
├── .env.example
└── README.md
```

## 安装步骤

```bash
cd E:\LaborLawRAG
python -m venv .venv && .venv\Scripts\activate     # 建议 Python 3.10/3.11
pip install -r requirements.txt
copy .env.example .env                             # 填入 OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL
python -m src.clean && python -m src.chunker       # 可选：重新生成清洗文本与切块
```

## 三条运行命令

```bash
python -m src.build_index        # 建索引（重跑会幂等重建 collection）
streamlit run app.py             # 旧版单用户界面（保留作调试）
python -m src.evaluate           # 跑评测；调试用 python -m src.evaluate --limit 5 --only A,C
python eval/run_extra.py         # 跑补充测试集（9 题，含工时类）；--testset 可指定其他集
python eval/run_multi.py         # 跑多轮场景（5 例：指代、回溯、换主题、元问题）
```

## Web 服务（多角色版）

```
uvicorn web.main:app --reload --port 8000     # 启动后访问 http://127.0.0.1:8000
```

- 两类角色：`user` 注册即用，只能问答与查看自己的会话；`admin` 用于维护知识库。
- 管理员账号由服务启动时创建，用户名/密码取自 `.env` 的 `ADMIN_USERNAME` / `ADMIN_PASSWORD`；
  未配置密码时会生成随机密码并打印到启动日志。
- 会话历史存 `data/app.sqlite3`（users / auth_tokens / chat_sessions / messages / documents）。
- 多轮记忆：最近 6 条消息进 Prompt，其中**助手回答只保留【结论】段**（完整答案动辄五六百字，四轮就吃满预算）；
  含指代词的追问（"那如果我月薪9000呢"）会自动回溯到"有主题的那一问"再拼接检索。
- 对话为**流式输出**：`POST /api/chat/stream` 用 SSE（`text/event-stream`）先推 `meta`（会话 id + 检索到的条文），
  再逐字推 `delta`，最后 `done`（耗时与引用校验）。非流式的 `/api/chat` 保留，评测与调试仍用它。
- 知识库管理页 `/admin`（仅管理员）：上传 txt/md/pdf，后台切块建索引；同名法规整部替换，
  删除时该法规全部条文一并移除。上传后 `src/index_ops.py` 会重建 BM25 并刷新检索器缓存。
  扫描版 PDF 无文本层，需先 OCR。
- 管理后台分三个标签页：**数据看板**（用户数、提问量、拒答率、引用可溯源率、14 天趋势、
  高频问题、知识库缺口）、**用户管理**（停用/启用、改角色、重置密码、删除）、**知识库**。
- 指标口径：线上**不叫"正确率"**——没有标准答案无法计算；用「引用可溯源率」（回答里的条号能否
  在检索结果中找到）+「拒答率」两个客观指标，质量基线看离线评测的 hit@3 / recall@5。
- 用户管理有自保护：不能停用/删除自己，至少保留一个可用管理员；停用会立即清空令牌强制下线。

## 常见问题

- **模型下载失败**：命令前设置镜像 `set HF_ENDPOINT=https://hf-mirror.com`，或把 `EMBED_MODEL` 指向本地模型目录。
- **chroma 版本冲突**：先 `pip install -U chromadb` 再 `pip install "opentelemetry-exporter-otlp-proto-grpc==<与 opentelemetry-proto 同版本>"`，仍冲突就删掉 `index/chroma` 重建。
- **中文乱码**：所有文件读写都显式 `encoding="utf-8"`，Windows 控制台可先执行 `chcp 65001`。

## 说明

- 条号一律保留中文数字原文（如"第四十七条"），切块与引用校验均以中文条号为唯一标识。
- `data/raw` 为空时 `build_index` 会自动生成 3 条【示例数据】条文，保证流程可跑通。
- 检索侧带同义词扩展（`src/query_expand.py`）：把"加班费/加班"映射为法条用语"延长工作时间、工资报酬"后再做 BM25 打分，法条原文不改。
- 界面默认"智能检索"（`src/cite_expand.py`）：固定走混合召回，再把命中条文里"依照本法第X条"引用的条文自动补进上下文——用户不需要理解向量/BM25 的区别。向量、BM25、RRF 三种单路保留在下拉框里，仅作排障用。
- `中华人民共和国劳动合同法` 为全本 98 条（含 2012 年修正），其余 4 部仍为便于演示的摘录，不构成法律意见。
