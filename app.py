import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jieba.posseg as pseg
import pandas as pd
import streamlit as st
from pypinyin import Style, lazy_pinyin


def _is_streamlit_cloud() -> bool:
    """
    判断是否运行在 Streamlit Community Cloud。
    """
    return bool(os.getenv("STREAMLIT_SHARING_MODE"))


if _is_streamlit_cloud():
    os.system("git lfs pull")


DEFAULT_DB_PATH = Path("textbook_corpus.db")

# 批量模式每页显示词数
BATCH_PAGE_SIZE = 150

# 终端检索模式：学科拼音 -> 中文（与 tabs 顺序一致）
SUBJECT_LABELS = {
    "yuwen": "语文",
    "shuxue": "数学",
    "kexue": "科学",
    "daofa": "道法",
}
TAB_ORDER = ["语文", "数学", "科学", "道法"]


@st.cache_resource
def get_cached_connection(db_path_str: str) -> sqlite3.Connection:
    """单例数据库连接，避免重复打开 DB 文件。Streamlit 重跑可能换线程，故允许跨线程使用。"""
    conn = sqlite3.connect(db_path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_connection(db_path: Path) -> sqlite3.Connection:
    return get_cached_connection(str(db_path))


def backup_database(db_path: Path) -> None:
    if not db_path.exists():
        return
    backup_dir = db_path.parent / "db_backups"
    backup_dir.mkdir(exist_ok=True)
    existing_backups = sorted(backup_dir.glob(f"{db_path.stem}_backup_*.db"))
    suffix = len(existing_backups) + 1
    backup_path = backup_dir / f"{db_path.stem}_backup_{suffix:03d}.db"
    shutil.copy2(db_path, backup_path)


def ensure_token_columns(conn: sqlite3.Connection) -> None:
    """
    若数据库缺少网站使用的兼容字段，自动迁移添加。

    新版语料库使用 is_poly 保存多义词标记；迁移到网站结构时同步到
    is_polysemy，并将 progress_score 同步为旧界面使用的 absolute_level，
    避免数据库更新后丢失已有标注或排序信息。
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(tokens);")
    cols = {row["name"] for row in cur.fetchall()}

    if "is_polysemy" not in cols:
        cur.execute(
            "ALTER TABLE tokens ADD COLUMN is_polysemy INTEGER NOT NULL DEFAULT 0;"
        )
        if "is_poly" in cols:
            cur.execute("UPDATE tokens SET is_polysemy = is_poly;")
    if "semantic_tag" not in cols:
        cur.execute("ALTER TABLE tokens ADD COLUMN semantic_tag TEXT;")
    if "word_semantics" not in cols:
        cur.execute("ALTER TABLE tokens ADD COLUMN word_semantics TEXT;")
    if "context_tag" not in cols:
        cur.execute("ALTER TABLE tokens ADD COLUMN context_tag TEXT;")
    if "word_semantics_json" not in cols:
        cur.execute("ALTER TABLE tokens ADD COLUMN word_semantics_json TEXT;")

    cur.execute("PRAGMA table_info(texts);")
    text_cols = {row["name"] for row in cur.fetchall()}
    if "absolute_level" not in text_cols:
        cur.execute("ALTER TABLE texts ADD COLUMN absolute_level INTEGER;")
        if "progress_score" in text_cols:
            cur.execute("UPDATE texts SET absolute_level = progress_score;")

    conn.commit()


def parse_word_semantics_json(raw: Optional[str]) -> Dict[str, Dict[str, str]]:
    """
    解析 word_semantics_json，格式 {"学科": {"semantics": "义1;义2", "pos": "v"}, ...}。
    返回 dict，缺键时返回 {}。
    """
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return {k: dict(v) if isinstance(v, dict) else {"semantics": "", "pos": str(v)} for k, v in data.items()}
    except (json.JSONDecodeError, TypeError):
        return {}


def merge_subject_into_semantics_json(
    current_raw: Optional[str], subject: str, semantics: str, pos: str
) -> str:
    """将某学科的 semantics/pos 合并进现有 JSON 字符串，返回新 JSON 字符串。"""
    data = parse_word_semantics_json(current_raw)
    data[subject] = {"semantics": (semantics or "").strip(), "pos": (pos or "").strip()}
    return json.dumps(data, ensure_ascii=False)


def ensure_indexes(conn: sqlite3.Connection) -> None:
    """建立检索索引，显著提升查询速度。"""
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_word ON tokens(word);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_is_verified ON tokens(is_verified);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_id ON sentences(id);")
    conn.commit()


def level_label(grade: int, term: int) -> str:
    term_map = {1: "上册", 2: "下册"}
    return f"{grade}年级{term_map.get(term, f'{term}册')}"


def level_str(version: str, grade: int, term: int) -> str:
    term_cn = "上" if term == 1 else "下"
    return f"[{version}] {grade}年级{term_cn}册"


def highlight_word(content: str, word: str) -> str:
    if not word or word not in content:
        return content
    return content.replace(word, f"**:red[{word}]**")


def search_by_word(conn: sqlite3.Connection, word: str) -> List[Dict[str, Any]]:
    """
    终端同款全匹配检索：
    返回 subject/version/grade/term/title/content/absolute_level/sentence_id/text_id/sentence_index。
    """
    if not word or not word.strip():
        return []
    word = word.strip()
    sql = """
    SELECT
        tx.subject,
        tx.version,
        tx.grade,
        tx.term,
        tx.title,
        s.content,
        tx.absolute_level,
        s.id   AS sentence_id,
        s.text_id,
        s.sentence_index
    FROM tokens t
    JOIN sentences s ON t.sentence_id = s.id
    JOIN texts tx ON s.text_id = tx.id
    WHERE t.word = ?
    ORDER BY tx.subject, tx.absolute_level ASC
    """
    cur = conn.execute(sql, (word,))
    return [dict(row) for row in cur.fetchall()]


def get_overall_stats(conn: sqlite3.Connection) -> Dict[str, float]:
    """
    返回总词汇量、已校对 / 待校对数量及进度。
    """
    cur = conn.cursor()

    cur.execute("SELECT COUNT(DISTINCT word) AS cnt FROM tokens;")
    total = cur.fetchone()["cnt"] or 0

    cur.execute(
        "SELECT COUNT(DISTINCT word) AS cnt FROM tokens WHERE is_verified = 1;"
    )
    verified = cur.fetchone()["cnt"] or 0

    pending = max(total - verified, 0)
    progress = (verified / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "verified": verified,
        "pending": pending,
        "progress": progress,
    }


def query_token_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    词频统计表：用于侧边栏音序导航和模糊检索。
    """
    sql = """
    SELECT
        word,
        COUNT(*) AS freq,
        SUM(is_verified) AS verified_count
    FROM tokens
    GROUP BY word
    ORDER BY freq DESC
    """
    df = pd.read_sql_query(sql, conn)
    return df


@st.cache_data(ttl=None)
def get_cached_word_list(
    db_path_str: str,
    only_unverified: bool,
    search_keyword: str,
) -> List[Tuple[str, int, int, str]]:
    """
    缓存「词 + 频次 + 已校对数 + 首字母」列表；删除/合并后需 st.cache_data.clear() 刷新。
    返回 [(word, freq, verified_count, initial), ...]，按 freq 降序。
    """
    conn = get_cached_connection(db_path_str)
    sql = """
    SELECT word, COUNT(*) AS freq, SUM(is_verified) AS verified_count
    FROM tokens
    GROUP BY word
    ORDER BY freq DESC
    """
    df = pd.read_sql_query(sql, conn)
    if only_unverified:
        df = df[df["verified_count"] == 0]
    if search_keyword and search_keyword.strip():
        df = df[df["word"].str.contains(search_keyword.strip(), na=False)]
    df["initial"] = df["word"].apply(get_initial_letter)
    return list(
        zip(
            df["word"].tolist(),
            df["freq"].astype(int).tolist(),
            df["verified_count"].astype(int).tolist(),
            df["initial"].tolist(),
        )
    )


def clear_word_list_cache() -> None:
    """删除/合并/拆分/确认校对后调用，强制刷新音序词表缓存。"""
    get_cached_word_list.clear()


def get_initial_letter(word: str) -> str:
    if not word:
        return "#"
    py = lazy_pinyin(word[0], style=Style.FIRST_LETTER)
    if not py:
        return "#"
    ch = py[0].upper()
    if "A" <= ch <= "Z":
        return ch
    return "#"


def get_word_profile(conn: sqlite3.Connection, word: str) -> Optional[Dict]:
    """
    获取该词的代表性标注：词性、多义标记、word_semantics；含 word_semantics_json（若有）。
    """
    sql = """
    SELECT word, pos, is_polysemy, word_semantics, word_semantics_json
    FROM tokens
    WHERE word = ?
    LIMIT 1
    """
    row = conn.execute(sql, (word,)).fetchone()
    if not row:
        return None
    js = ""
    if "word_semantics_json" in row.keys() and row["word_semantics_json"]:
        js = row["word_semantics_json"]
    return {
        "word": row["word"],
        "pos": row["pos"] or "",
        "is_polysemy": bool(row["is_polysemy"]),
        "word_semantics": row["word_semantics"] or "",
        "word_semantics_json": js,
    }


def get_subjects_for_word(conn: sqlite3.Connection, word: str) -> List[str]:
    """该词在库中出现的学科列表（去重，顺序稳定）。"""
    sql = """
    SELECT DISTINCT t.subject
    FROM tokens k
    JOIN sentences s ON k.sentence_id = s.id
    JOIN texts t ON s.text_id = t.id
    WHERE k.word = ?
    ORDER BY t.subject
    """
    rows = conn.execute(sql, (word,)).fetchall()
    return [r["subject"] for r in rows]


def get_subject_semantics_for_word(
    conn: sqlite3.Connection, word: str
) -> Dict[str, Dict[str, str]]:
    """从任意一条该词的 token 的 word_semantics_json 解析出各学科的 semantics/pos，用于预填。"""
    sql = """
    SELECT word_semantics_json FROM tokens WHERE word = ? AND word_semantics_json IS NOT NULL AND word_semantics_json != ''
    LIMIT 1
    """
    row = conn.execute(sql, (word,)).fetchone()
    if not row:
        return {}
    return parse_word_semantics_json(row["word_semantics_json"])


def sync_subject_semantics(
    conn: sqlite3.Connection,
    word: str,
    subject: str,
    semantics: str,
    pos: str,
) -> int:
    """
    将该词在该学科下的语义/词性同步到所有同名且属于该学科的 token 记录；
    合并进现有 word_semantics_json，并置 is_verified=1。返回更新行数。
    """
    cur = conn.cursor()
    # 取一条该词+该学科的 token 的当前 JSON，合并后写回所有该词+该学科的 token
    cur.execute(
        """
        SELECT word_semantics_json FROM tokens k
        JOIN sentences s ON k.sentence_id = s.id
        JOIN texts t ON s.text_id = t.id
        WHERE k.word = ? AND t.subject = ?
        LIMIT 1
        """,
        (word, subject),
    )
    row = cur.fetchone()
    current_raw = row["word_semantics_json"] if row else None
    new_json = merge_subject_into_semantics_json(current_raw, subject, semantics, pos)

    cur.execute(
        """
        UPDATE tokens
        SET word_semantics_json = ?, is_verified = 1
        WHERE word = ?
          AND sentence_id IN (
            SELECT s.id FROM sentences s JOIN texts t ON s.text_id = t.id WHERE t.subject = ?
          )
        """,
        (new_json, word, subject),
    )
    affected = cur.rowcount
    conn.commit()
    return affected


def delete_word_globally(conn: sqlite3.Connection, word: str) -> int:
    """
    从 tokens 表中彻底删除该词的所有记录。
    """
    backup_database(DEFAULT_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM tokens WHERE word = ?;", (word,))
    affected = cur.rowcount
    conn.commit()
    return affected


def batch_delete_words(
    conn: sqlite3.Connection, words: List[str], db_path: Path
) -> int:
    """
    批量物理删除：在事务内执行 DELETE FROM tokens WHERE word IN (...)。
    执行前自动备份到 db_backups 并生成同目录下的 .db_bak 临时备份；返回删除的总行数。
    """
    if not words:
        return 0
    backup_database(db_path)
    # 同目录临时备份，便于快速回滚
    bak_path = db_path.parent / (db_path.name + ".bak")
    if db_path.exists():
        shutil.copy2(db_path, bak_path)
    print(f"[批量删除] 即将删除 {len(words)} 个词条，已自动备份数据库（含 {bak_path}）。")
    cur = conn.cursor()
    try:
        conn.execute("BEGIN")
        placeholders = ",".join("?" * len(words))
        cur.execute(
            f"DELETE FROM tokens WHERE word IN ({placeholders});",
            words,
        )
        total = cur.rowcount
        conn.commit()
        return total
    except Exception as e:
        conn.rollback()
        raise e


def is_non_chinese_word(word: str) -> bool:
    """
    判断是否包含数字、英文或常见标点（。，！？、—等），用于“选中所有非中文字符”。
    """
    if not word or not word.strip():
        return True
    # 包含数字、ASCII 字母、常见标点即视为“非纯中文”
    pattern = re.compile(
        r"[0-9A-Za-z\]\[。，！？、—…·\"\'\'（）【】\s\.\,\!\?\;\:\-\*\/\\]"
    )
    return bool(pattern.search(word))


def save_word_semantics_globally(
    conn: sqlite3.Connection,
    word: str,
    word_semantics: str,
    is_polysemy: bool,
) -> None:
    """
    将词级语义（word_semantics）与多义标记应用到该词所有记录。
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE tokens
        SET word_semantics = ?, is_polysemy = ?
        WHERE word = ?
        """,
        (word_semantics or None, 1 if is_polysemy else 0, word),
    )
    conn.commit()


def query_sentences_by_word(conn: sqlite3.Connection, word: str) -> pd.DataFrame:
    """
    按 absolute_level 升序获取该词的所有例句及上下文所需元数据。
    """
    sql = """
    SELECT
        k.id           AS token_id,
        k.word         AS word,
        k.pos          AS pos,
        k.is_verified  AS is_verified,
        k.is_polysemy  AS is_polysemy,
        k.semantic_tag AS semantic_tag,
        k.context_tag  AS context_tag,
        k.word_semantics AS word_semantics,

        s.id           AS sentence_id,
        s.content      AS sentence_content,
        s.para_index   AS para_index,
        s.sentence_index AS sentence_index,

        t.id           AS text_id,
        t.version      AS version,
        t.subject      AS subject,
        t.grade        AS grade,
        t.term         AS term,
        t.lesson_num   AS lesson_num,
        t.title        AS title,
        t.absolute_level AS absolute_level
    FROM tokens k
    JOIN sentences s ON k.sentence_id = s.id
    JOIN texts t ON s.text_id = t.id
    WHERE k.word = ?
    ORDER BY t.absolute_level ASC, s.sentence_index ASC
    """
    return pd.read_sql_query(sql, conn, params=(word,))


def query_context(
    conn: sqlite3.Connection, text_id: int, center_sentence_index: int, window: int = 2
) -> List[Dict]:
    sql = """
    SELECT id, content, sentence_index
    FROM sentences
    WHERE text_id = ?
      AND sentence_index BETWEEN ? AND ?
    ORDER BY sentence_index ASC
    """
    rows = conn.execute(
        sql,
        (
            text_id,
            max(1, center_sentence_index - window),
            center_sentence_index + window,
        ),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "sentence_index": r["sentence_index"],
            "content": r["content"],
        }
        for r in rows
    ]


def update_sentence_context_tag(
    conn: sqlite3.Connection,
    token_id: int,
    context_tag: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE tokens
        SET context_tag = ?, is_verified = 1
        WHERE id = ?
        """,
        (context_tag or None, token_id),
    )
    conn.commit()


def apply_context_to_subject(
    conn: sqlite3.Connection,
    word: str,
    subject: str,
    context_tag: str,
) -> int:
    """
    将某个 context_tag 一键应用到该学科中所有该词的记录。
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE tokens
        SET context_tag = ?, is_verified = 1
        WHERE word = ?
          AND sentence_id IN (
            SELECT s.id
            FROM sentences s
            JOIN texts t ON s.text_id = t.id
            WHERE t.subject = ?
          )
        """,
        (context_tag or None, word, subject),
    )
    affected = cur.rowcount
    conn.commit()
    return affected


def goto_next_word() -> None:
    """
    将 current_word_index 前进一位，用于“连打模式”。
    """
    word_list = st.session_state.get("word_list") or []
    if not word_list:
        return
    idx = st.session_state.get("current_word_index", 0)
    if idx < len(word_list) - 1:
        st.session_state["current_word_index"] = idx + 1


def merge_tokens_in_sentence(
    conn: sqlite3.Connection,
    sentence_id: int,
    base_token_id: int,
    merged_word: str,
) -> bool:
    """
    多合一：在指定句子中，以 base_token 为中心，
    找到若干连续 token，使其拼接后等于 merged_word。
    删除这些旧 token，插入一个新的 merged_word token（is_verified=0）。
    """
    merged_word = merged_word.strip()
    if not merged_word:
        return False

    cur = conn.cursor()
    cur.execute(
        "SELECT id, word FROM tokens WHERE sentence_id = ? ORDER BY id ASC;",
        (sentence_id,),
    )
    tokens = cur.fetchall()
    ids = [r["id"] for r in tokens]
    words = [r["word"] for r in tokens]

    if base_token_id not in ids:
        return False

    base_idx = ids.index(base_token_id)

    # 最多向两侧扩展若干个 token（上限设为 6，避免过长组合）
    n = len(tokens)
    found_span = None
    max_span = min(6, n)

    for span_len in range(1, max_span + 1):
        for start in range(0, n - span_len + 1):
            end = start + span_len  # [start, end)
            if not (start <= base_idx < end):
                continue
            candidate = "".join(words[start:end])
            if candidate == merged_word:
                found_span = (start, end)
                break
        if found_span:
            break

    if not found_span:
        return False

    start, end = found_span
    delete_ids = ids[start:end]

    # 删除旧 token
    cur.execute(
        f"DELETE FROM tokens WHERE id IN ({','.join('?' for _ in delete_ids)});",
        delete_ids,
    )

    # 新 token 的词性自动标注
    pos = None
    for w in pseg.cut(merged_word):
        pos = w.flag
        break

    cur.execute(
        """
        INSERT INTO tokens (sentence_id, word, pos, is_verified, is_polysemy)
        VALUES (?, ?, ?, 0, 0)
        """,
        (sentence_id, merged_word, pos),
    )
    conn.commit()
    return True


def split_token(
    conn: sqlite3.Connection,
    token_id: int,
    sentence_id: int,
    split_text: str,
) -> bool:
    """
    一拆多：删除原有长词 token_id，按输入的空格分隔词列表，
    在该句中插入多个新 token（is_verified=0，词性用 jieba.posseg 自动标注）。
    """
    parts = [p.strip() for p in split_text.split() if p.strip()]
    if not parts:
        return False

    cur = conn.cursor()
    # 删除原 token
    cur.execute("DELETE FROM tokens WHERE id = ?;", (token_id,))

    # 为每个子词生成词性并插入
    for part in parts:
        pos = None
        for w in pseg.cut(part):
            pos = w.flag
            break
        cur.execute(
            """
            INSERT INTO tokens (sentence_id, word, pos, is_verified, is_polysemy)
            VALUES (?, ?, ?, 0, 0)
            """,
            (sentence_id, part, pos),
        )

    conn.commit()
    return True


def main() -> None:
    st.set_page_config(
        page_title="小学教材语料库科研系统（专家版）",
        layout="wide",
    )

    st.title("小学教材语料库科研系统（专家版）")

    # --- 数据库设置 ---
    with st.sidebar:
        st.header("数据库设置")
        db_path_str = st.text_input(
            "SQLite 数据库路径",
            value=str(DEFAULT_DB_PATH),
        )
        db_path = Path(db_path_str)
        if not db_path.exists():
            st.warning("数据库文件不存在，请先运行 main.py 完成入库。")
            return

    conn = get_connection(db_path)
    ensure_token_columns(conn)
    ensure_indexes(conn)

    st.markdown("## 终端同款检索")
    search_word = st.text_input(
        "输入词汇（全匹配）",
        placeholder="输入要检索的词，如：实验、单位",
        key="search_input_terminal_like",
    )
    if search_word and search_word.strip():
        rows = search_by_word(conn, search_word)
        if not rows:
            st.warning(f"未找到与「{search_word}」全匹配的例句。")
        else:
            # 学科分布统计（保持与 search_term.py 一致）
            df_all = pd.DataFrame(rows)
            subject_pinyin = df_all["subject"].tolist()
            freq_map = {}
            for lab in TAB_ORDER:
                pinyin = next(k for k, v in SUBJECT_LABELS.items() if v == lab)
                freq_map[lab] = subject_pinyin.count(pinyin)
            chart_df = pd.DataFrame(
                [{"学科": k, "频次": v} for k, v in freq_map.items()]
            ).set_index("学科")
            st.subheader("学科分布")
            st.bar_chart(chart_df)

            # 按学科分组（保持与 search_term.py 一致）
            by_subject = {lab: [] for lab in TAB_ORDER}
            for r in rows:
                lab = SUBJECT_LABELS.get(r["subject"], r["subject"])
                if lab in by_subject:
                    by_subject[lab].append(r)

            tabs = st.tabs(TAB_ORDER)
            for tab, label in zip(tabs, TAB_ORDER):
                with tab:
                    group = by_subject[label]
                    if not group:
                        st.caption("该学科下无例句。")
                        continue
                    for r in group:
                        meta = (
                            f"{level_str(r['version'], r['grade'], r['term'])} | "
                            f"《{r['title'] or ''}》"
                        )
                        content_highlighted = highlight_word(r["content"], search_word)
                        st.caption(meta)
                        with st.expander(
                            f"例句：{r['content'][:60]}{'…' if len(r['content']) > 60 else ''}"
                        ):
                            st.markdown(content_highlighted)
                            ctx_list = query_context(
                                conn,
                                text_id=r["text_id"],
                                center_sentence_index=r["sentence_index"],
                                window=2,
                            )
                            if ctx_list:
                                st.caption("上下文（前后各 2 句）：")
                                for ctx in ctx_list:
                                    idx = ctx["sentence_index"]
                                    text = ctx["content"]
                                    if idx == r["sentence_index"]:
                                        text = highlight_word(text, search_word)
                                        st.markdown(f"**👉 {idx}：{text}**")
                                    else:
                                        st.caption(f"{idx}：{text}")

    st.markdown("---")

    # --- 操作计数与统计看板节流（每 10 次操作或手动刷新才重算）---
    if "op_count" not in st.session_state:
        st.session_state.op_count = 0
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = get_overall_stats(conn)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 2, 1])
    with col1:
        st.metric("总词汇量（去重）", st.session_state.last_stats["total"])
    with col2:
        st.metric("已校对词汇数", st.session_state.last_stats["verified"])
    with col3:
        st.metric("待校对词汇数", st.session_state.last_stats["pending"])
    with col4:
        st.metric("校对进度", f"{st.session_state.last_stats['progress']:.2f}%")
        st.progress(
            st.session_state.last_stats["progress"] / 100.0
            if st.session_state.last_stats["progress"] > 0
            else 0.0
        )
    with col5:
        if st.button("🔄 刷新看板", key="refresh_stats_btn"):
            st.session_state.last_stats = get_overall_stats(conn)
            st.rerun()

    # --- 侧边栏：二级导航（字母 -> 下拉选词），不渲染海量词条 ---
    with st.sidebar:
        st.header("音序导航 / 过滤")
        only_unverified = st.checkbox("仅看未校对词", value=False)
        search_keyword = st.text_input("模糊搜索词汇（包含）", value="")

        letter_options = ["全部"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        selected_letter = st.selectbox("按首字母筛选", letter_options, key="letter_select")

    # 使用缓存获取轻量词表（仅存当前字母下的 word 列表，不存 DataFrame）
    cached_tuples = get_cached_word_list(
        str(db_path),
        only_unverified,
        search_keyword.strip(),
    )
    # 按字母过滤
    if selected_letter != "全部":
        cached_tuples = [t for t in cached_tuples if t[3] == selected_letter]
    word_list = [t[0] for t in cached_tuples]
    freq_map = {t[0]: t[1] for t in cached_tuples}
    verified_map = {t[0]: t[2] for t in cached_tuples}

    # 初始化 / 更新 session_state：只存当前字母的 word_list（轻量）
    if "word_list" not in st.session_state:
        st.session_state.word_list = word_list
        st.session_state.current_word_index = 0
    else:
        if word_list != st.session_state.word_list:
            st.session_state.word_list = word_list
            st.session_state.current_word_index = 0

    if not word_list:
        selected_word = None
    else:
        idx = st.session_state.get("current_word_index", 0)
        if idx < 0:
            idx = 0
        if idx >= len(word_list):
            idx = len(word_list) - 1
        st.session_state.current_word_index = idx
        st.session_state.word_list = word_list
        selected_word = word_list[idx]

    # 批量管理模式开关（在词汇列表上方）
    if "batch_mode" not in st.session_state:
        st.session_state.batch_mode = False
    if "batch_selected" not in st.session_state:
        st.session_state.batch_selected: Set[str] = set()

    with st.sidebar:
        batch_mode = st.checkbox(
            "批量管理模式",
            value=st.session_state.batch_mode,
            key="batch_mode_cb",
        )
        st.session_state.batch_mode = batch_mode

        if not word_list:
            st.info("当前筛选条件下没有词汇。")
        else:
            if batch_mode:
                # ---------- 批量操作面板（分页，每页 BATCH_PAGE_SIZE 个词）----------
                st.markdown("**批量操作**")
                if "batch_page" not in st.session_state:
                    st.session_state.batch_page = 0
                total_pages = max(1, (len(word_list) + BATCH_PAGE_SIZE - 1) // BATCH_PAGE_SIZE)
                start_idx = st.session_state.batch_page * BATCH_PAGE_SIZE
                end_idx = min(start_idx + BATCH_PAGE_SIZE, len(word_list))
                page_words = word_list[start_idx:end_idx]

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("全选", key="batch_select_all"):
                        st.session_state.batch_selected = set(word_list)
                        st.rerun()
                with b2:
                    if st.button("反选", key="batch_invert"):
                        st.session_state.batch_selected = set(word_list) - st.session_state.batch_selected
                        st.rerun()
                with b3:
                    if st.button("选中非中文字符", key="batch_select_non_cn"):
                        st.session_state.batch_selected = {
                            w for w in word_list if is_non_chinese_word(w)
                        }
                        st.rerun()

                st.caption(f"第 {st.session_state.batch_page + 1}/{total_pages} 页，本页 {len(page_words)} 个词；已选 **{len(st.session_state.batch_selected)}** / {len(word_list)}")

                # 仅渲染当前页的复选框
                COLS_PER_ROW = 2
                for start in range(0, len(page_words), COLS_PER_ROW):
                    row_words = page_words[start : start + COLS_PER_ROW]
                    cols = st.columns(COLS_PER_ROW)
                    for col, w in zip(cols, row_words):
                        with col:
                            label = f"{w} (频次:{freq_map.get(w, 0)})"
                            current = w in st.session_state.batch_selected
                            key = f"batch_cb_{w}_{start_idx}_{start}"
                            st.checkbox(label, value=current, key=key)

                new_selected: Set[str] = set()
                for start in range(0, len(page_words), COLS_PER_ROW):
                    for w in page_words[start : start + COLS_PER_ROW]:
                        key = f"batch_cb_{w}_{start_idx}_{start}"
                        if st.session_state.get(key, False):
                            new_selected.add(w)
                # 当前页以复选框为准；其他页保留原选中
                st.session_state.batch_selected = (
                    st.session_state.batch_selected - set(page_words)
                ) | new_selected

                prev_p, next_p = st.columns(2)
                with prev_p:
                    if st.button("上一页", key="batch_prev_page") and st.session_state.batch_page > 0:
                        st.session_state.batch_page -= 1
                        st.rerun()
                with next_p:
                    if st.button("下一页", key="batch_next_page") and st.session_state.batch_page < total_pages - 1:
                        st.session_state.batch_page += 1
                        st.rerun()

                if st.button("🗑 批量删除选中词汇", type="primary", key="batch_delete_btn"):
                    to_delete = list(st.session_state.batch_selected)
                    if not to_delete:
                        st.warning("请先勾选要删除的词汇。")
                    else:
                        backup_database(db_path)
                        total = batch_delete_words(conn, to_delete, db_path)
                        clear_word_list_cache()
                        st.session_state.op_count = st.session_state.get("op_count", 0) + 1
                        if st.session_state.op_count % 10 == 0:
                            st.session_state.last_stats = get_overall_stats(conn)
                        st.session_state.batch_selected = set()
                        st.session_state.batch_page = 0
                        st.session_state.current_word_index = 0
                        st.success(f"已批量删除 {total} 条 token 记录。列表将刷新。")
                        st.rerun()
                selected_word = None
            else:
                # ---------- 二级导航：下拉框选词，不渲染几千个按钮 ----------
                st.write(f"当前筛选共 **{len(word_list)}** 个词")
                labels = [
                    f"{'✅' if verified_map.get(w, 0) > 0 else '❓'} {w}"
                    for w in word_list
                ]
                selected_idx = st.selectbox(
                    "词汇列表（点击可回溯修正）",
                    options=list(range(len(word_list))),
                    index=st.session_state.current_word_index if word_list else 0,
                    format_func=lambda i: labels[i],
                    key="word_selectbox",
                )
                if selected_idx != st.session_state.current_word_index:
                    st.session_state.current_word_index = selected_idx
                    selected_word = word_list[selected_idx]

        if selected_word:
            st.markdown("---")
            st.markdown("⚠ **危险操作：清理非词汇字符**")
            if st.button("彻底删除此词（从 tokens 中移除）"):
                affected = delete_word_globally(conn, selected_word)
                clear_word_list_cache()
                st.session_state.op_count = st.session_state.get("op_count", 0) + 1
                if st.session_state.op_count % 10 == 0:
                    st.session_state.last_stats = get_overall_stats(conn)
                st.success(f"已从 tokens 表中删除 {affected} 条记录。")
                goto_next_word()
                st.rerun()

    st.markdown("---")

    if not selected_word:
        if st.session_state.get("batch_mode", False):
            st.info(
                "当前为 **批量管理模式**。请在左侧勾选要删除的词汇（可先用「选中非中文字符」快速筛选），"
                "再点击「批量删除选中词汇」。删除后列表会自动刷新，进度归零到当前字母第一个词。"
            )
        else:
            st.info("请在左侧选择一个词开始校对。")
        return

    # --- 词级语义与标注 ---
    profile = get_word_profile(conn, selected_word)
    if not profile:
        st.warning("数据库中未找到该词的记录，请检查。")
        return

    st.subheader(f"当前词：**{selected_word}** — 分学科语义与历时例句")

    # 导航按钮
    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button("⬅ 上一个词"):
            if st.session_state.get("word_list"):
                idx = st.session_state.get("current_word_index", 0)
                if idx > 0:
                    st.session_state["current_word_index"] = idx - 1
            st.rerun()
    with nav_next:
        if st.button("跳过 / 下一个 ➡"):
            goto_next_word()
            st.rerun()

    col_left, col_right = st.columns([2, 3])

    # ---------- 左侧：分学科语义标注 ----------
    with col_left:
        subjects = get_subjects_for_word(conn, selected_word)
        subject_semantics = get_subject_semantics_for_word(conn, selected_word)

        st.markdown("**分学科语义标注**（多义项用 ； 分隔）")

        for subj in subjects:
            info = subject_semantics.get(subj, {})
            if isinstance(info, dict):
                sem_val = info.get("semantics", "") or ""
                pos_val = info.get("pos", "") or ""
            else:
                sem_val = pos_val = ""

            with st.expander(f"**{subj}**", expanded=(len(subjects) <= 2)):
                pos_input = st.text_input(
                    f"词性（{subj}）",
                    value=pos_val,
                    key=f"pos_{selected_word}_{subj}",
                )
                sem_input = st.text_area(
                    "语义标注（多义项用 ； 分隔）",
                    value=sem_val,
                    key=f"sem_{selected_word}_{subj}",
                    height=80,
                )
                if st.button(
                    "同步该学科语义至所有同名记录",
                    key=f"sync_subj_{selected_word}_{subj}",
                ):
                    backup_database(db_path)
                    n = sync_subject_semantics(
                        conn, selected_word, subj, sem_input, pos_input
                    )
                    clear_word_list_cache()
                    st.session_state.op_count = st.session_state.get("op_count", 0) + 1
                    if st.session_state.op_count % 10 == 0:
                        st.session_state.last_stats = get_overall_stats(conn)
                    st.success(f"已同步「{subj}」下 {n} 条记录，并设为已校对。")
                    st.rerun()

        st.markdown("---")
        if st.button("确认校对并跳到下一个词", type="primary", key="confirm_and_next"):
            backup_database(db_path)
            cur = conn.cursor()
            cur.execute(
                "UPDATE tokens SET is_verified = 1 WHERE word = ?;",
                (selected_word,),
            )
            conn.commit()
            clear_word_list_cache()
            st.session_state.op_count = st.session_state.get("op_count", 0) + 1
            if st.session_state.op_count % 10 == 0:
                st.session_state.last_stats = get_overall_stats(conn)
            goto_next_word()
            st.rerun()

    # ---------- 右侧：历时例句（按学科分组，组内按 absolute_level 升序）----------
    with col_right:
        df_sent = query_sentences_by_word(conn, selected_word)
        if df_sent.empty:
            st.info("该词暂无例句。")
            return

        # 按学科分组，组内按 absolute_level、sentence_index 排序
        df_sent = df_sent.sort_values(
            ["subject", "absolute_level", "sentence_index"]
        )
        st.markdown("**历时例句（按学科分组，组内按年级升序）**")

        for subj in df_sent["subject"].unique():
            sub_df = df_sent[df_sent["subject"] == subj]
            st.markdown(f"### {subj}")
            for _, row in sub_df.iterrows():
                prefix = (
                    f"[{row['version']}][{row['subject']}] "
                    f"{level_label(row['grade'], row['term'])} | {row['title'] or ''}"
                )
                exp_label = f"{prefix}：{row['sentence_content']}"

                with st.expander(exp_label):
                    st.markdown("**上下文（前后各 2 句）**")
                    ctx_rows = query_context(
                        conn,
                        text_id=row["text_id"],
                        center_sentence_index=row["sentence_index"],
                        window=2,
                    )
                    for ctx in ctx_rows:
                        marker = "👉 " if ctx["sentence_index"] == row["sentence_index"] else ""
                        st.write(f"{marker}{ctx['sentence_index']}：{ctx['content']}")

                    st.markdown("---")
                    st.markdown("**句/学科级 context_tag（可选）**")
                    default_ctx = row["context_tag"] or ""
                    context_tag_val = st.text_input(
                        "在该语境下的精确含义",
                        value=default_ctx,
                        key=f"context_{row['token_id']}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(
                            "仅保存当前句的 context_tag",
                            key=f"btn_sentence_{row['token_id']}",
                        ):
                            backup_database(db_path)
                            update_sentence_context_tag(
                                conn,
                                token_id=row["token_id"],
                                context_tag=context_tag_val.strip(),
                            )
                            st.success("已保存。")
                            st.rerun()
                    with c2:
                        if st.button(
                            "应用到该学科全部例句",
                            key=f"btn_subject_{row['token_id']}",
                        ):
                            backup_database(db_path)
                            affected = apply_context_to_subject(
                                conn,
                                word=selected_word,
                                subject=row["subject"],
                                context_tag=context_tag_val.strip(),
                            )
                            st.success(f"已应用于「{row['subject']}」下 {affected} 条记录。")
                            st.rerun()

                    st.markdown("---")
                    st.markdown("**分词纠错**")
                    merge_input = st.text_input(
                        "多合一：合并为本句中的一个新词（如：科学）",
                        key=f"merge_{row['token_id']}",
                    )
                    split_input = st.text_input(
                        "一拆多：拆成多个词，空格分隔（如：中国 科学）",
                        key=f"split_{row['token_id']}",
                    )
                    c3, c4 = st.columns(2)
                    with c3:
                        if st.button(
                            "执行多合一（本句）",
                            key=f"btn_merge_{row['token_id']}",
                        ):
                            backup_database(db_path)
                            ok = merge_tokens_in_sentence(
                                conn,
                                sentence_id=row["sentence_id"],
                                base_token_id=row["token_id"],
                                merged_word=merge_input,
                            )
                            if ok:
                                st.success("多合一成功，已重置为待核对。")
                                clear_word_list_cache()
                                st.session_state.op_count = st.session_state.get("op_count", 0) + 1
                                if st.session_state.op_count % 10 == 0:
                                    st.session_state.last_stats = get_overall_stats(conn)
                            else:
                                st.error("无法在本句中找到能拼成该词的一组连续分词。")
                            if ok:
                                goto_next_word()
                            st.rerun()
                    with c4:
                        if st.button(
                            "执行一拆多（本句）",
                            key=f"btn_split_{row['token_id']}",
                        ):
                            backup_database(db_path)
                            ok = split_token(
                                conn,
                                token_id=row["token_id"],
                                sentence_id=row["sentence_id"],
                                split_text=split_input,
                            )
                            if ok:
                                st.success("一拆多成功，已重置为待核对。")
                                clear_word_list_cache()
                                st.session_state.op_count = st.session_state.get("op_count", 0) + 1
                                if st.session_state.op_count % 10 == 0:
                                    st.session_state.last_stats = get_overall_stats(conn)
                            else:
                                st.error("请输入至少一个要拆分出的词。")
                            if ok:
                                goto_next_word()
                            st.rerun()


if __name__ == "__main__":
    main()
