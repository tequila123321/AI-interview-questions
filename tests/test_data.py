from agentkit.data.chunking import chunk_fixed, chunk_markdown, chunk_recursive
from agentkit.data.cleaning import Doc, clean_documents, normalize_text
from agentkit.tokens import estimate_tokens

FOOTER = "版权所有 © 示例公司 | 点击查看更多"


def test_normalize_nfkc_and_whitespace():
    assert normalize_text("ＡＢＣ　１２３  \r\n\r\n\r\n下一段​") == "ABC 123\n\n下一段"


def test_clean_pipeline_reports_every_decision():
    policy = (
        "退款政策：签收后 7 天内可申请退款，需商品完好、配件齐全、不影响二次销售；生鲜、定制类商品不支持无理由退款，质量问题除外。"
        "退款到账时间：审核通过后原路退回，银行卡 3-5 个工作日，第三方支付 1-3 个工作日；如超时未到账请联系客服并提供订单号。"
    )
    docs = [
        Doc("a", f"{policy}\n{FOOTER}"),
        Doc("b", f"{policy}\n{FOOTER}"),                                   # 精确重复
        Doc("c", f"{policy}（2026 年版）\n{FOOTER}"),                          # 近似重复（爬到的另一个版本）
        Doc("d", f"配送说明：一般 2-3 天送达，偏远地区 5-7 天。联系电话 13812345678。\n{FOOTER}"),
        Doc("e", f"短\n{FOOTER}"),
        Doc("f", FOOTER),
    ]
    kept, rep = clean_documents(docs, min_chars=10)
    assert rep.input_docs == 6 and rep.output_docs == 2
    assert FOOTER in rep.boilerplate_lines and rep.boilerplate_removed == 6
    assert rep.dropped_exact_dup == 1 and rep.dropped_near_dup == 1
    assert rep.dropped_short == 1 and rep.dropped_empty == 1
    assert rep.pii_redactions == 1 and "13812345678" not in kept[1].text
    assert all("content_hash" in d.meta for d in kept)


def test_fixed_chunks_cover_text_with_overlap():
    doc = Doc("d", "甲" * 100 + "乙" * 100 + "丙" * 100)
    chunks = chunk_fixed(doc, size_tokens=120, overlap_tokens=20)
    assert chunks[0].text.startswith("甲") and chunks[-1].text.endswith("丙")
    assert chunks[1].text[:20] == chunks[0].text[-20:]        # 重叠
    assert all(c.tokens <= 120 for c in chunks)


def test_recursive_chunks_respect_size_and_prefer_sentence_boundaries():
    text = "。".join(f"这是第{i}句话，用来测试递归切分是否会在句子边界断开" for i in range(30)) + "。"
    chunks = chunk_recursive(Doc("d", text), size_tokens=100)
    assert all(c.tokens <= 110 for c in chunks)
    assert all(c.text.endswith("。") for c in chunks[:-1])


def test_markdown_chunks_carry_heading_path_and_parent_child():
    md = "# 退款政策\n\n## 一般情况\n签收后 7 天内可申请。\n\n## 特殊情况\n" + "生鲜商品不支持无理由退款，但质量问题可申请。" * 20
    chunks = chunk_markdown(Doc("kb", md), size_tokens=80)
    parents = [c for c in chunks if c.meta["role"] == "parent"]
    children = [c for c in chunks if c.meta["role"] == "child"]
    assert [p.meta["heading_path"] for p in parents] == [["退款政策", "一般情况"], ["退款政策", "特殊情况"]]
    assert children and all(c.meta["parent_id"] == parents[1].id for c in children)
    assert all(c.text.startswith("退款政策 > 特殊情况") for c in children)
    assert estimate_tokens(parents[1].text) > 80 and all(c.tokens <= 100 for c in children)
