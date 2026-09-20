"""记忆层：抽取、召回、压缩、上下文预算、"第 20 轮丢信息"诊断。"""

from agentkit.harness import build_agent
from agentkit.llm.base import Message, ToolCall
from agentkit.memory.compressor import ExtractiveSummarizer
from agentkit.memory.context_builder import build_context
from agentkit.memory.diagnose import diagnose_missing_fact
from agentkit.memory.extractor import RuleFactExtractor
from agentkit.memory.manager import MemoryConfig, MemoryManager
from agentkit.memory.tiers import Fact, LongTermMemory
from agentkit.observability.trace import Tracer

FILLER = ["今天上海天气怎么样", "退款政策是什么", "谢谢", "北京天气呢", "多久能到", "好的"]


def test_rule_extractor_finds_hard_constraints():
    facts = RuleFactExtractor().extract("我叫王小明，订单号 SO-000123，对花生过敏，预算 500 元，地址是上海市浦东新区张江路1号，不要香菜", 1)
    got = {f.key: (f.value, f.pinned) for f in facts}
    assert got["姓名"] == ("王小明", True)
    assert got["订单号"] == ("SO-123", True)          # 归一化
    assert got["过敏"] == ("花生", True)
    assert got["预算"][0].startswith("500")
    assert got["地址"][0].startswith("上海市浦东新区")
    assert got["忌口"] == ("香菜", False)


def test_ltm_pinned_always_recalled_and_multi_values_coexist():
    ltm = LongTermMemory()
    ltm.upsert(Fact("订单号", "SO-1", 1, pinned=True, multi=True))
    ltm.upsert(Fact("订单号", "SO-2", 3, pinned=True, multi=True))
    ltm.upsert(Fact("忌口", "香菜", 2))
    ltm.upsert(Fact("预算", "500", 2))
    ltm.upsert(Fact("预算", "800", 5))                       # 覆盖
    keys = {f.key: f.value for f in ltm.all()}
    assert keys["订单号"] == "SO-1" and keys["订单号#2"] == "SO-2" and keys["预算"] == "800"
    recalled = ltm.recall("我不吃香菜", k=2, current_turn=6)
    assert [f.key for f in recalled][:2] == ["订单号", "订单号#2"]     # pinned 先
    assert "忌口" in [f.key for f in recalled] and "预算" not in [f.key for f in recalled]


def test_compression_keeps_pinned_facts_and_traces_what_was_dropped():
    mm = MemoryManager(config=MemoryConfig(working_high_water_tokens=80, keep_recent_turns=2, summary_max_tokens=60), tracer=Tracer())
    mm.start_turn("我的订单号是 SO-000123，我对花生过敏")
    mm.append(Message.assistant("好的"))
    mm.end_turn()
    for t in ["今天天气不错啊，聊聊别的", "我想了解一下你们的退款政策具体是怎样的", "谢谢你的解释"]:
        mm.start_turn(t)
        mm.append(Message.assistant("好的，" + t))
        report = mm.end_turn()
    assert report is not None and report.dropped_turns == [2]              # 第 1 轮在更早一次压缩里已被压掉
    assert "SO-123" in mm.summary and "花生" in mm.summary
    assert all(m.meta["turn"] >= 3 for m in mm.working)
    assert mm.tracer.find("memory.compress")[0]["dropped_turns"] == [1]


def test_naive_summarizer_loses_facts_when_not_told_to_keep():
    s = ExtractiveSummarizer(clip_chars=10, keep_facts=False)
    msgs = [Message.user("我的订单号是 SO-000123，我对花生过敏", turn=1)]
    out = s.summarize("", msgs, ["SO-123", "花生"], 100)
    assert "花生" not in out


def test_context_builder_trims_oldest_and_never_splits_tool_groups():
    tc = ToolCall("c1", "lookup_order", {"order_id": "SO-1"})
    working = [
        Message.user("第一轮很长的问题 " * 20, turn=1),
        Message.assistant("", [tc], turn=1),
        Message.tool(tc, '{"ok": true}', turn=1),
        Message.assistant("答一", turn=1),
        Message.user("第二轮", turn=2),
        Message.assistant("", [tc], turn=2),
        Message.tool(tc, '{"ok": true}', turn=2),
        Message.assistant("答二", turn=2),
        Message.user("第三轮", turn=3),
    ]
    ctx = build_context("sys", "", [], working, budget_tokens=120)
    assert ctx.messages[0].role == "user" and ctx.messages[0].tool_call_id is None
    assert ctx.report.dropped_turns == [1] and ctx.report.included_turns == [2, 3]
    roles = [m.role for m in ctx.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]


def test_context_builder_pinned_facts_do_not_compete_with_budget():
    facts = [Fact("订单号", "SO-123", 1, pinned=True)]
    working = [Message.user("x" * 400, turn=1), Message.user("y", turn=2)]
    ctx = build_context("sys", "", facts, working, budget_tokens=60)
    assert "SO-123" in ctx.system and ctx.report.pinned_keys == ["订单号"]


def _run_twenty_turns(profile: str):
    a = build_agent(profile=profile)
    a.run_turn("先记一下：我的订单号是 SO-000123，另外我对花生过敏，不要香菜")
    for i in range(18):
        a.run_turn(FILLER[i % len(FILLER)])
    r = a.run_turn("我的订单号是多少？")
    return a, r


def test_turn_20_naive_forgets_and_diagnosis_points_to_write_side():
    a, r = _run_twenty_turns("naive")
    assert r.turn == 20 and "不记得" in r.text
    d = diagnose_missing_fact(a.tracer.events, "SO-000123", at_turn=20)
    assert d.stage == "not_extracted"


def test_turn_20_guarded_remembers_via_pinned_recall():
    a, r = _run_twenty_turns("guarded")
    assert r.turn == 20 and "SO-123" in r.text
    ctx = a.tracer.find("context.built", turn=20)[0]
    assert "订单号" in ctx["pinned_keys"] and 1 not in ctx["included_turns"]           # 原文早被压缩掉，靠关键信息块
    assert any(1 in e["dropped_turns"] for e in a.tracer.find("memory.compress"))
    d = diagnose_missing_fact(a.tracer.events, "SO-123", at_turn=20, aliases=["SO-000123"])
    assert d.stage == "present_in_context"


def test_diagnosis_distinguishes_not_recalled_for_unpinned_fact():
    a, _ = _run_twenty_turns("guarded")
    d = diagnose_missing_fact(a.tracer.events, "香菜", at_turn=20)
    assert d.stage == "not_recalled"


def test_diagnosis_never_stated():
    a, _ = _run_twenty_turns("guarded")
    assert diagnose_missing_fact(a.tracer.events, "SO-777", at_turn=20).stage == "never_stated"
