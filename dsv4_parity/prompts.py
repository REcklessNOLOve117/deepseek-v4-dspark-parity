from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .io import sha256_json


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    category: str
    text: str

    def to_dict(self) -> dict[str, str]:
        result = asdict(self)
        result["text_sha256"] = sha256_json(self.text)
        return result


PROMPTS: tuple[Prompt, ...] = (
    Prompt("code_01", "code", "Write a Python function that returns the first non-repeating character in a Unicode string. Explain the complexity."),
    Prompt("code_02", "code", "Implement binary search in Rust for a sorted slice of i64 values. Return Option<usize> and include two tests."),
    Prompt("code_03", "code", "Find and fix the bug in this code, then explain it:\n\n```python\ndef chunks(xs, n):\n    for i in range(0, len(xs), n):\n        yield xs[i:i+n-1]\n```"),
    Prompt("code_04", "code", "Design a SQL query that returns each customer's most recent paid order from tables customers(id) and orders(id, customer_id, paid_at, status)."),
    Prompt("code_05", "code", "Write a TypeScript function to topologically sort a directed acyclic graph represented as Map<string, string[]>. Detect cycles."),
    Prompt("code_06", "code", "Explain why this C loop may overflow and provide a safe rewrite: for (int i = 0; i <= n; ++i) sum += a[i];"),
    Prompt("code_07", "code", "Create a regular expression and a short Python example that extracts ISO dates in YYYY-MM-DD format but rejects 2025-13-40."),
    Prompt("code_08", "code", "Given an LRU cache API get(key) and put(key, value), outline an O(1) implementation and state its invariants."),
    Prompt("tool_01", "tool", "You may call get_weather(city, date). The user asks: What will the weather be in Shanghai tomorrow? Respond with exactly one JSON tool call and no prose."),
    Prompt("tool_02", "tool", "Available tool: search_docs(query: string). Find documentation about rotating API keys. Emit a tool call using the schema {\"name\": string, \"arguments\": object}."),
    Prompt("tool_03", "tool", "Available tools are add(a,b) and multiply(a,b). Compute (17+25)*3 by issuing the necessary tool calls in dependency order."),
    Prompt("tool_04", "tool", "Use calendar_create(title, start_iso, duration_minutes) for: Schedule 'design review' at 2026-09-18 14:30 Asia/Shanghai for 45 minutes. Output JSON only."),
    Prompt("tool_05", "tool", "Tool schema: read_file({path:string, line?:integer}). Read line 120 of /srv/app/config.py. Do not invent the file contents."),
    Prompt("tool_06", "tool", "You can call stock_quote(ticker, market). Request the current quote for NVIDIA on the US market using one compact JSON object."),
    Prompt("tool_07", "tool", "Available tool send_email(to, subject, body). Draft a call that asks alice@example.com to review PR 42 by Friday; preserve all facts and add none."),
    Prompt("tool_08", "tool", "The only tool is database_query(sql). Query the count of active users created after 2026-01-01. Return a syntactically valid tool call."),
    Prompt("zh_01", "chinese_qa", "请用三句话解释为什么天空通常是蓝色的，并说明日落时颜色变化的原因。"),
    Prompt("zh_02", "chinese_qa", "北京和上海都位于中国东部。请简要比较两地冬季气候，不要编造精确温度。"),
    Prompt("zh_03", "chinese_qa", "什么是数据库事务的 ACID？请分别给出一句通俗解释。"),
    Prompt("zh_04", "chinese_qa", "把下面这句话改得更简洁，但保留原意：由于目前现阶段资源相对比较有限，因此我们暂时先不开展全面测试。"),
    Prompt("zh_05", "chinese_qa", "解释“相关性不等于因果性”，并给出一个不涉及医疗的例子。"),
    Prompt("zh_06", "chinese_qa", "一台服务每秒处理 80 个请求，平均每个请求耗时 100 毫秒。仅凭这些信息能否判断并发数？说明理由。"),
    Prompt("zh_07", "chinese_qa", "请列出代码评审时最值得优先检查的四类问题，每类不超过十个字。"),
    Prompt("zh_08", "chinese_qa", "用户说“系统偶尔变慢”。你会先收集哪五项信息？按诊断价值排序。"),
    Prompt("reason_01", "reasoning", "A box contains 3 red, 4 blue, and 5 green balls. Two balls are drawn without replacement. What is the probability they have the same color? Show the calculation."),
    Prompt("reason_02", "reasoning", "All bloops are razzies. No razzies are tazzies. Can any bloop be a tazzy? Give a one-sentence justification."),
    Prompt("reason_03", "reasoning", "A train travels 120 km at 60 km/h and returns 120 km at 40 km/h. What is its average speed for the round trip?"),
    Prompt("reason_04", "reasoning", "There are three switches downstairs and one light bulb upstairs. You may go upstairs only once. How can you identify the controlling switch?"),
    Prompt("reason_05", "reasoning", "Sequence: 2, 6, 12, 20, 30. Give the next two terms and state the rule you used."),
    Prompt("reason_06", "reasoning", "A meeting has six people. If every pair shakes hands exactly once, how many handshakes occur? Derive the result."),
    Prompt("reason_07", "reasoning", "You have 9 identical-looking coins; one is heavier. Find it in two balance-scale weighings and explain the branches."),
    Prompt("reason_08", "reasoning", "If a fair six-sided die is rolled until a 6 appears, what is the expected number of rolls? Explain briefly."),
)


def validate_prompts(prompts: Iterable[Prompt] = PROMPTS) -> None:
    rows = tuple(prompts)
    if len(rows) != 32:
        raise ValueError(f"expected 32 prompts, got {len(rows)}")
    ids = [row.prompt_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1
        if not row.text.strip():
            raise ValueError(f"empty prompt: {row.prompt_id}")
    expected = {"code": 8, "tool": 8, "chinese_qa": 8, "reasoning": 8}
    if counts != expected:
        raise ValueError(f"unexpected category counts: {counts}")


validate_prompts()

