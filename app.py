# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="生命力比對系統",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# 工具函式
# =========================================================
def clean_col(c):
    if not isinstance(c, str): c = str(c)
    c = unicodedata.normalize("NFKC", c)
    return c.replace("\u3000", "").replace("\xa0", "").strip()

def norm_tax(x):
    if pd.isna(x): return np.nan
    s = unicodedata.normalize("NFKC", str(x)).strip()
    if "." in s: s = s.split(".", 1)[0]
    digits = re.sub(r"\D", "", s)
    if 1 <= len(digits) <= 8: return digits.zfill(8)
    return np.nan

def norm_name(x):
    if pd.isna(x): return np.nan
    s = unicodedata.normalize("NFKC", str(x)).strip().lower()
    s = s.replace("臺", "台")
    s = re.sub(r"[\s\u3000]", "", s)
    return s or np.nan

CN_NUM = {"零":"0","一":"1","二":"2","三":"3","四":"4",
          "五":"5","六":"6","七":"7","八":"8","九":"9","〇":"0"}
SUFFIX_PAT = r"(股份有限(責任)?公司|有限公司|有限責任公司|分公司|公司|協會|學會|促進會|基金會|文教基金會|社會企業|法人|工作室|協進會)$"

def normalize_digits(s):
    for k, v in CN_NUM.items(): s = s.replace(k, v)
    return s.replace("Ⅰ","I").replace("Ⅱ","II").replace("Ⅲ","III").replace("Ⅳ","IV").replace("Ⅴ","V")

def norm_name_strong(x):
    if pd.isna(x): return np.nan
    s = unicodedata.normalize("NFKC", str(x)).strip().lower()
    s = s.replace("臺", "台")
    s = re.sub(r"[^\w\u4e00-\u9fa5]", "", s)
    s = normalize_digits(s)
    for _ in range(2): s = re.sub(SUFFIX_PAT, "", s)
    return s.strip() or np.nan

def pick(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def coalesce(*vals):
    for v in vals:
        if pd.notna(v) and str(v).strip() not in ("", "nan", "NaN", "None", "<NA>"):
            return v
    return np.nan

def find_tax_col_from_cols(cols):
    for c in cols:
        if "統" in unicodedata.normalize("NFKC", str(c)) and "編" in unicodedata.normalize("NFKC", str(c)):
            return c
    return None

def find_tax_col(df):
    for c in ["統一編號", "統編", "統一編號(統編)"]:
        if c in df.columns: return c
    fuzzy = [c for c in df.columns if "統" in c and "編" in c]
    return fuzzy[0] if fuzzy else None

def list_len7(df, col=None):
    if col is None: col = find_tax_col(df)
    if col is None or col not in df.columns: return pd.DataFrame()
    src = df[col].astype("string")
    return df.loc[src.str.replace(r"\D", "", regex=True).str.len() == 7, [col]].copy()

def pad8_list(df, col=None):
    if col is None: col = find_tax_col(df)
    if col is None or col not in df.columns:
        return pd.DataFrame(columns=["原始統編", "補零後統編"])
    src = df[col].astype("string")
    padded = src.map(norm_tax)
    out = pd.DataFrame({"原始統編": src, "補零後統編": padded})
    mask = src.notna() & padded.notna() & (src.str.replace(r"\D", "", regex=True).str.len().fillna(0).astype(int) < 8)
    return out[mask].copy()

def safe_series(df, col_name, dtype="string"):
    return df[col_name] if col_name in df.columns else pd.Series(index=df.index, dtype=dtype)

def fb(r, new_col, *old_cols):
    val = r.get(new_col)
    if pd.notna(val) and str(val).strip() not in ("", "nan", "NaN", "None", "<NA>"):
        return val
    for c in old_cols:
        v = r.get(c)
        if pd.notna(v) and str(v).strip() not in ("", "nan", "NaN", "None", "<NA>"):
            return v
    return np.nan

def make_excel(sheets):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, index=False, sheet_name=name[:31])
    return buf.getvalue()

# =========================================================
# Step 1：文章去重
# =========================================================
def run_dedup(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "article_title" not in df.columns:
        raise ValueError("找不到 'article_title' 欄位")
    if "url" not in df.columns:
        raise ValueError("找不到 'url' 欄位")

    df["article_title"] = df["article_title"].astype(str).str.strip().str.lower()
    df["url"] = df["url"].astype(str).str.strip().str.lower()

    dupe_stats = (
        df.groupby(["article_title", "url"])
        .size()
        .reset_index(name="duplicate_count")
        .sort_values(by="duplicate_count", ascending=False)
    )
    df_unique = df.drop_duplicates(subset=["article_title", "url"], keep="first")
    df_dupes  = dupe_stats[dupe_stats["duplicate_count"] > 1]

    return df_unique, df_dupes, len(df)

# =========================================================
# Step 2：文字探勘（網路爬蟲 + jieba）
# =========================================================
def _make_session():
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    s = requests.Session()
    retries = Retry(
        total=3, connect=2, read=2, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    s.headers.update({"User-Agent": UA})
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _fetch_one(session, url, timeout=5):
    import html as htmllib
    from bs4 import BeautifulSoup
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        try:
            if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "ascii"):
                resp.encoding = resp.apparent_encoding or "utf-8"
        except Exception:
            pass
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        title = ""
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
        else:
            og = soup.find("meta", property="og:title")
            if og and og.get("content"):
                title = og.get("content").strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        all_text = soup.get_text("\n")
        all_text = re.sub(r"\n{2,}", "\n", all_text).strip()
        return title, all_text, ""
    except Exception as e:
        return "", "", repr(e)

def _parse_date(text):
    import html as htmllib
    s = htmllib.unescape(str(text))
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", s)
    if m: return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m: return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    m = re.search(r"(20\d{6})", s)
    if m: return m.group(1)
    return None

ORG_KEYWORDS = [
    "協會","基金會","中心","學會","學校","大學","研究院","公司",
    "球隊","俱樂部","體育館","公園","委員會","工作室","團隊",
    "中華隊","奧運","帕運","亞帕運","公開賽","平台","計畫","計劃"
]
ORG_REGEX   = re.compile(r"([一-龥A-Za-z0-9．／\-\s]{2,20}?(?:" + "|".join(ORG_KEYWORDS) + r"))")
PLAN_REGEX  = re.compile(r"([一-龥A-Za-z0-9]{2,20}(?:計畫|方案|行動|策略|平台))")

def _extract_persons(text):
    names  = re.findall(r"([一-龥]{2,3})(?:(?:表示|提到|說|指出|加入|帶領))", text or "")
    names += re.findall(r"([一-龥]{2,3})：「", text or "")
    names += re.findall(r"(?:學員|選手|教練|記者)([一-龥]{2,3})", text or "")
    blacklist = {"台北","臺北","台灣","臺灣","巴黎","雅典","澳洲","新北","板橋"}
    out, seen = [], set()
    for n in names:
        if n not in blacklist and n not in seen:
            out.append(n); seen.add(n)
    return out

def _extract_orgs(text):
    orgs = ORG_REGEX.findall(text or "")
    out, seen = [], set()
    for o in orgs:
        o2 = re.sub(r"\s+", "", o)
        if o2 and o2 not in seen:
            out.append(o2); seen.add(o2)
    return out

def _extract_plans(text):
    plans = PLAN_REGEX.findall(text or "")
    out, seen = [], set()
    for p in plans:
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def _jieba_keywords(text, topk=15, use_textrank=True):
    import jieba.analyse
    if not text:
        return []
    tfidf = jieba.analyse.extract_tags(text, topK=topk, withWeight=False) or []
    tr = []
    if use_textrank:
        try:
            tr = jieba.analyse.textrank(text, topK=topk, withWeight=False) or []
        except Exception:
            tr = []
    combined = list(dict.fromkeys(tfidf + tr))
    return [w.strip() for w in combined if len(w.strip()) >= 2]

def run_text_mining(df, max_workers=6, timeout_sec=5, topk=15, use_textrank=True,
                    progress_cb=None):
    """
    df 需有 url 欄位，article_title 欄位選填。
    progress_cb: callable(pct: float, msg: str)
      爬取階段佔 0–0.5，jieba 解析階段佔 0.5–1.0
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import jieba  # noqa: ensure jieba is loaded

    def _prog(pct, msg):
        if progress_cb:
            progress_cb(min(pct, 1.0), msg)

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "url" not in df.columns:
        raise ValueError("找不到 'url' 欄位")
    if "article_title" not in df.columns:
        df["article_title"] = ""

    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    urls = df["url"].astype(str).str.strip().tolist()
    n = len(urls)

    # ── 階段一：併發爬取（0% → 50%）──
    _prog(0, f"爬取中 (0 / {n})...")
    session = _make_session()
    res_title = [None] * n
    res_text  = [None] * n
    res_err   = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_fetch_one, session, u, timeout_sec): i
                      for i, u in enumerate(urls)}
        done = 0
        for fut in as_completed(future_map):
            i = future_map[fut]
            res_title[i], res_text[i], res_err[i] = fut.result()
            done += 1
            _prog(done / n * 0.5, f"爬取中 ({done} / {n})...")

    # ── 階段二：jieba 解析（50% → 100%）──
    _prog(0.5, f"jieba 解析中 (0 / {n})...")
    parsed_dates, final_titles = [], []
    jieba_cols, rule_cols, merged_cols = [], [], []
    content_len_col, raw_preview_col   = [], []

    for i, u in enumerate(urls):
        raw_title  = str(df.at[i, "article_title"] or "").strip()
        page_title = res_title[i] or ""
        text_full  = res_text[i]  or ""
        final_title = raw_title or page_title
        final_titles.append(final_title)

        date_str = (_parse_date(text_full) or _parse_date(final_title) or
                    _parse_date(u) or "unknown")
        parsed_dates.append(date_str)

        full_text_for_kw = f"{final_title}\n{text_full}"
        jieba_kw = _jieba_keywords(full_text_for_kw, topk=topk, use_textrank=use_textrank)
        persons  = _extract_persons(full_text_for_kw)
        orgs     = _extract_orgs(full_text_for_kw)
        plans    = _extract_plans(full_text_for_kw)

        rule_kw = list(dict.fromkeys(persons + orgs + plans))
        merged  = list(dict.fromkeys(jieba_kw + rule_kw))

        jieba_cols.append("、".join(jieba_kw))
        rule_cols.append("、".join(rule_kw))
        merged_cols.append("、".join(merged))
        content_len_col.append(len(text_full))
        raw_preview_col.append(text_full[:120].replace("\n", " "))

        _prog(0.5 + (i + 1) / n * 0.5, f"jieba 解析中 ({i + 1} / {n})...")

    df["article_title"] = final_titles
    df["parsed_date"]   = parsed_dates

    counters, article_ids = {}, []
    for date_str in df["parsed_date"]:
        counters[date_str] = counters.get(date_str, 0) + 1
        article_ids.append(f"vita_{date_str}_{counters[date_str]}")
    df["article_id"]    = article_ids
    df["jieba_keywords"] = jieba_cols
    df["rule_keywords"]  = rule_cols
    df["keywords"]       = merged_cols
    df["fetch_error"]    = res_err
    df["content_chars"]  = content_len_col
    df["raw_preview"]    = raw_preview_col

    return df

# =========================================================
# Step 4：文章比對（關鍵字 × 組織名稱）
# =========================================================
_COMPANY_SUFFIXES = ["股份有限公司", "有限公司", "公司"]
_SEP_PAT  = re.compile(r"[;；、，,\-/—－–／/·\.\s]+")
_CH_PAT   = re.compile(r"^[\u4e00-\u9fff]$")
_PAREN_PAT = re.compile(r"^(?P<main>.*?)(?:\(|（)(?P<brand>.*?)(?:\)|）)\s*$")

def _has_cjk(s):
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

def _is_valid_token(t):
    t = str(t or "").strip()
    if not t: return False
    if _has_cjk(t): return len(re.sub(r"[^\u4e00-\u9fff]", "", t)) >= 2
    s = t.replace(" ", "")
    if len(s) < 3: return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_. ]*", t))

def _filter_tokens(iterable):
    out = set()
    for x in (iterable or []):
        x = str(x or "").strip()
        if _is_valid_token(x): out.add(x)
    return out

def _strip_company_suffix(name):
    s = str(name)
    for suf in _COMPANY_SUFFIXES:
        if s.endswith(suf): return s[:-len(suf)]
    return s

def _norm_name_match(name):
    return str(name).lower().replace(" ", "").replace("\u3000", "")

def _expand_name_variants(raw):
    out = set()
    if not raw: return out
    base = str(raw).strip()
    candidates = {base, _strip_company_suffix(base)}
    m = _PAREN_PAT.match(base)
    if m:
        main = (m.group("main") or "").strip()
        brand = (m.group("brand") or "").strip()
        if main:
            candidates.add(main); candidates.add(_strip_company_suffix(main))
        if brand:
            candidates.add(brand)
            for t in _SEP_PAT.split(brand):
                if t: candidates.add(t.strip())
    tokens = [t for t in _SEP_PAT.split(base) if t]
    if 2 <= len(tokens) <= 3 and all(_CH_PAT.match(t) and len(t) == 1 for t in tokens):
        candidates.add("".join(tokens))
        candidates.add("".join(reversed(tokens)))
    return _filter_tokens(candidates)

def _parse_existing_links(old_val):
    pairs = []
    if not old_val: return pairs
    s = str(old_val).strip()
    if s.lower() in ("無", "nan", "none", ""): return pairs
    seen = set()
    for p in re.split(r"[；;]+", s):
        p = p.strip()
        if not p: continue
        m = re.search(r"(?:\(|（)?(?P<year>20\d{2})(?:\)|）|年)?\s*(?P<url>http\S+)", p)
        if m: y, u = m.group("year") or "", m.group("url").strip()
        else: y, u = "", p.strip()
        if u and u not in seen:
            pairs.append((y, u)); seen.add(u)
    return pairs

def _assemble_links(pairs):
    uniq = {}
    for y, u in pairs:
        if u and u not in uniq: uniq[u] = y
    items = [(y, u) for u, y in uniq.items()]
    def _sk(t):
        y, u = t
        return (0, int(y), u) if re.fullmatch(r"20\d{2}", y or "") else (1, 9999, u)
    items.sort(key=_sk)
    return "；".join([f"({y}){u}" if re.fullmatch(r"20\d{2}", y or "") else u for y, u in items])

def _union_tokens(old_text, new_tokens):
    bag = set()
    if old_text:
        bag |= _filter_tokens(re.split(r"[；;]+", str(old_text).strip()))
    bag |= _filter_tokens(new_tokens)
    return "；".join(sorted(bag, key=lambda x: (len(x), x))) if bag else ""

def _clean_cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

def run_article_match(kw_df, df_main, new_date_int=0, prev_kw_map=None,
                      match_fields=None, progress_cb=None):
    """
    kw_df   : Step 2 輸出（含 keywords, url, parsed_date, article_id）
    df_main : Step 3 輸出的 02_精簡表 工作表
    new_date_int : 「新文章」門檻（int，如 20251001），0 表示全部視為新文章
    prev_kw_map  : {組織名稱: 舊關鍵字字串} 來自上一期
    match_fields : 比對欄位清單，預設 ["組織名稱2","負責人2","單位聯絡人_new"]
    progress_cb  : callable(pct, msg)
    """
    if match_fields is None:
        match_fields = ["組織名稱2", "負責人2", "單位聯絡人_new"]
    if prev_kw_map is None:
        prev_kw_map = {}

    def _prog(pct, msg):
        if progress_cb: progress_cb(min(pct, 1.0), msg)

    df_main = df_main.copy()
    df_main.columns = pd.Index(df_main.columns).map(
        lambda s: str(s or "").replace("\u3000", " ").strip())

    for c in ["生命力新聞", "生命力新聞_new_match", "比對到的關鍵字", "比對狀態"]:
        if c not in df_main.columns:
            df_main[c] = ""

    # 準備關鍵字 DataFrame
    kw = kw_df.copy()
    kw.columns = [c.strip().lower() for c in kw.columns]
    kw["kw_nospace"] = (kw.get("keywords", pd.Series(dtype=str))
                         .astype(str).fillna("").str.lower()
                         .str.replace(" ", "", regex=False)
                         .str.replace("\u3000", "", regex=False))
    kw["dateint"] = pd.to_numeric(kw.get("parsed_date", pd.Series(dtype=str)),
                                  errors="coerce").astype("Int64")
    kw_new = kw[kw["dateint"].ge(new_date_int).fillna(False)].copy() if new_date_int else kw

    from pandas import Series as _S
    def safe_col(col):
        return df_main[col].astype(str).fillna("") if col in df_main.columns else _S([""] * len(df_main), index=df_main.index)

    col_newcat = safe_col("新增分類")
    col_prev   = safe_col("上一期有無")
    prev_exists = col_prev.str.contains("有|是|Y|1", na=False)
    G1 = col_newcat.str.contains("持續存在", na=False)
    G2 = col_newcat.str.contains("初期後出現", na=False) & prev_exists
    G3 = col_newcat.str.contains("本期真正新增", na=False)
    G4 = col_newcat.str.contains("歷史補回", na=False)
    mask_merge_prev = G1 | G2 | G4

    targets_idx = list(df_main.index[G1 | G2 | G3 | G4])
    total = len(targets_idx)
    _prog(0, f"比對中 (0 / {total})...")

    for step_i, i in enumerate(targets_idx):
        row = df_main.loc[i]
        use_all = bool(G3.loc[i]) or bool(G4.loc[i])
        kw_sub = kw if use_all else kw_new

        names = [_clean_cell(row.get(c)) for c in match_fields if c in df_main.columns]
        candidates = set()
        for name in names:
            candidates |= _expand_name_variants(name)

        pairs_new, hit_terms = set(), set()
        for orig in candidates:
            cand = _norm_name_match(orig)
            mask = kw_sub["kw_nospace"].str.contains(cand, na=False, regex=False)
            if mask.any():
                hit_terms.add(orig)
                for _, r_art in kw_sub.loc[mask, ["dateint", "url"]].iterrows():
                    y = str(int(r_art["dateint"]))[:4] if pd.notna(r_art["dateint"]) else ""
                    pairs_new.add((y, str(r_art["url"]).strip()))

        old_links_text = _clean_cell(row.get("生命力新聞"))
        existing_pairs = set(_parse_existing_links(old_links_text))
        old_urls = {u for _, u in existing_pairs}
        true_new_pairs = [(y, u) for (y, u) in pairs_new if u not in old_urls]

        df_main.at[i, "生命力新聞_new_match"] = _assemble_links(true_new_pairs)

        old_kw = _clean_cell(row.get("比對到的關鍵字"))
        merged_kw = _union_tokens(old_kw, hit_terms)
        if bool(mask_merge_prev.loc[i]) and prev_kw_map:
            key = _clean_cell(row.get("組織名稱2"))
            merged_kw = _union_tokens(merged_kw, _filter_tokens(
                re.split(r"[；;]+", prev_kw_map.get(key, ""))))

        if not pairs_new and not hit_terms and not merged_kw:
            merged_kw = "nan"
        df_main.at[i, "比對到的關鍵字"] = merged_kw

        if pairs_new:
            df_main.at[i, "比對狀態"] = "本期新增"
        elif existing_pairs:
            df_main.at[i, "比對狀態"] = "僅歷史"

        _prog((step_i + 1) / total * 0.8, f"比對中 ({step_i + 1} / {total})...")

    # 生成比對結果整理
    _prog(0.85, "生成比對結果整理...")
    records = []
    org_data = []
    for idx, r in df_main.loc[G1 | G2 | G3 | G4].iterrows():
        cands = set()
        for c in match_fields:
            if c in df_main.columns:
                cands |= _expand_name_variants(_clean_cell(r.get(c)))
        org_data.append(({_norm_name_match(x) for x in cands}, cands,
                          bool(G3.loc[idx]) or bool(G4.loc[idx])))

    for _, art in kw.iterrows():
        txt = str(art.get("kw_nospace") or "")
        dt = art.get("dateint")
        hit = set()
        for norm_set, orig_set, use_all in org_data:
            if not use_all and new_date_int and (pd.isna(dt) or dt < new_date_int):
                continue
            if any(n in txt for n in norm_set):
                hit |= {o for o in orig_set if _norm_name_match(o) in txt}
        kw_out = "；".join(sorted(_filter_tokens(hit), key=lambda x: (len(x), x)))
        if not hit: kw_out = "nan"
        records.append({
            "article_title": art.get("article_title", ""),
            "url":           art.get("url", ""),
            "parsed_date":   art.get("parsed_date", ""),
            "article_id":    art.get("article_id", ""),
            "keywords":      art.get("keywords", ""),
            "比對到的關鍵字": kw_out,
            "比對狀態":      "有" if hit else "無",
        })
    df_report = pd.DataFrame(records)

    _prog(1.0, "完成！")
    return df_main, df_report

# =========================================================
# Step 3：比對分析
# =========================================================
def _find_prev_col(P_prev, bare_col, new_col):
    for name in [bare_col, new_col]:
        if name in P_prev.columns:
            return name
    matches = [c for c in P_prev.columns if c.startswith(bare_col)]
    return matches[0] if matches else None

def _build_init_keys(df, tax_col=None, name_col=None):
    df = df.copy()
    df.columns = [clean_col(c) for c in df.columns]
    t = tax_col or pick(df, ["統一編號_final","統一編號_x","統一編號"]) or find_tax_col(df)
    n = name_col
    if n is None:
        n = "K_name" if "K_name" in df.columns else pick(df, ["組織名稱2_final","組織名稱2_x","組織名稱2","組織名稱","名稱"])
    df["_kt"] = safe_series(df, t).map(norm_tax) if t else pd.Series(index=df.index, dtype="string")
    if n == "K_name" and "K_name" in df.columns:
        df["_kn"] = df["K_name"].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    else:
        df["_kn"] = safe_series(df, n).map(norm_name) if n else pd.Series(index=df.index, dtype="string")
    keys = df["_kt"].fillna("") + "|" + df["_kn"].fillna("")
    mask = keys.str.strip().ne("|") & keys.str.strip().ne("")
    return set(keys[mask])

def run_analysis(A, B, P_prev, progress_cb=None):
    STEPS = 9
    def _prog(step, msg):
        if progress_cb:
            progress_cb(step / STEPS, msg)

    _prog(1, "正規化欄位...")
    A.columns = [clean_col(c) for c in A.columns]
    B.columns = [clean_col(c) for c in B.columns]

    tax_col_A = find_tax_col(A)
    tax_col_B = find_tax_col(B)
    if tax_col_A is None: raise KeyError(f"A 檔找不到統編欄位：{A.columns.tolist()}")
    if tax_col_B is None: raise KeyError(f"B 檔找不到統編欄位：{B.columns.tolist()}")

    name_col_A = "組織名稱2" if "組織名稱2" in A.columns else pick(A, ["組織名稱", "名稱"])
    name_col_B = "組織名稱2" if "組織名稱2" in B.columns else pick(B, ["組織名稱", "名稱"])
    if name_col_A is None: raise KeyError(f"A 找不到名稱欄：{A.columns.tolist()}")
    if name_col_B is None: raise KeyError(f"B 找不到名稱欄：{B.columns.tolist()}")

    A["統編_原始"] = A[tax_col_A]
    B["統編_原始"] = B[tax_col_B]
    A["K_tax"]   = A[tax_col_A].map(norm_tax)
    A["K_name"]  = A[name_col_A].map(norm_name)
    A["K_name2"] = A[name_col_A].map(norm_name_strong)
    B["K_tax"]   = B[tax_col_B].map(norm_tax)
    B["K_name"]  = B[name_col_B].map(norm_name)
    B["K_name2"] = B[name_col_B].map(norm_name_strong)

    # init_keys 固定從 B 檔（最初期比對結果）取得
    B["KEY_init"] = B["K_tax"].fillna("") + "|" + B["K_name"].fillna("")
    init_keys = set(B["KEY_init"])

    _prog(2, "建立上期 KEY 集合...")
    prev_keys = set()
    if P_prev is not None:
        P_prev.columns = [clean_col(c) for c in P_prev.columns]
        prev_tax_col  = pick(P_prev, ["統一編號_final", "統一編號_x", "統一編號"])
        prev_name_col = pick(P_prev, ["組織名稱2_final", "組織名稱2_x", "組織名稱2"])
        P_prev["K_tax_prev"]  = safe_series(P_prev, prev_tax_col).map(norm_tax) if prev_tax_col else pd.Series(index=P_prev.index, dtype="string")
        P_prev["K_name_prev"] = safe_series(P_prev, prev_name_col).map(norm_name) if prev_name_col else pd.Series(index=P_prev.index, dtype="string")
        if "K_name" in P_prev.columns:
            P_prev["K_name_prev"] = P_prev["K_name"].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        P_prev["KEY_prev"] = P_prev["K_tax_prev"].fillna("") + "|" + P_prev["K_name_prev"].fillna("")
        mask_prev = P_prev["KEY_prev"].str.strip().ne("|") & P_prev["KEY_prev"].str.strip().ne("")
        prev_keys = set(P_prev.loc[mask_prev, "KEY_prev"])

    A_len7 = list_len7(A); B_len7 = list_len7(B)
    A_pad  = pad8_list(A); B_pad  = pad8_list(B)

    _prog(3, "執行嚴格比對（inner join）...")
    A_keyed = A.dropna(subset=["K_tax", "K_name2"]).copy()
    B_keyed = B.dropna(subset=["K_tax", "K_name2"]).copy()
    keep_inner = ["K_tax", "K_name", "K_name2"]
    A_suf = A_keyed.rename(columns={c: f"{c}_new" for c in A_keyed.columns if c not in keep_inner})
    B_suf = B_keyed.rename(columns={c: f"{c}_old" for c in B_keyed.columns if c not in keep_inner})
    merged = A_suf.merge(B_suf, on=["K_tax", "K_name2"], how="inner")
    if "K_name_x" in merged.columns and "K_name" not in merged.columns:
        merged["K_name"] = merged["K_name_x"]

    A_id = pick(merged, [f"{tax_col_A}_new", "統一編號_new", "統一編號_x", "統一編號"])
    A_nm = pick(merged, [f"{name_col_A}_new", "組織名稱2_new", "組織名稱2_x", "組織名稱2"])
    B_id = pick(merged, [f"{tax_col_B}_old", "統一編號_old", "統一編號_y"])
    B_nm = pick(merged, [f"{name_col_B}_old", "組織名稱2_old", "組織名稱2_y"])
    if not all([A_id, A_nm, B_id, B_nm]):
        raise KeyError(f"無法識別統編/名稱欄：A_id={A_id}, A_nm={A_nm}")

    one2one = merged.sort_values(["K_tax", "K_name2"]).drop_duplicates(subset=[B_id, B_nm], keep="first")
    cols_03 = [c for c in [A_id, A_nm, "排序2_new", "組織類型_new", "負責人2_new",
        "單位聯絡電話(市話)_new", "單位聯絡電話-分機_new", "單位聯絡手機_new",
        "電子信箱Email_new", "單位聯絡人_new", "縣市_new", "地址_new",
        "商品分類一_new", "商品服務一_new", "商品分類二_new", "商品服務二_new",
        "商品分類三_new", "商品服務三_new",
        "聯合國永續發展目標-1_new", "聯合國永續發展目標-2_new", "聯合國永續發展目標-3_new",
        "官方網站_new", "Facebook_new", "影音網址_new", "電子商務_new",
        "編號2_old", "組織名稱_old", "負責人2_old"] if c in one2one.columns]
    final_df = one2one[cols_03].copy().rename(columns={A_id: "統一編號", A_nm: "組織名稱2"})
    many_to_one = merged.copy()

    _prog(4, "執行全量外連接（outer join）...")
    keep_outer = ["K_tax", "K_name"]
    A_tag = A.copy().rename(columns={c: f"{c}_new" for c in A.columns if c not in keep_outer})
    B_tag = B.copy().rename(columns={c: f"{c}_old" for c in B.columns if c not in keep_outer})
    for col in ["K_name", "K_tax"]:
        A_tag[col] = A_tag[col].astype(object)
        B_tag[col] = B_tag[col].astype(object)

    B_nokey = B_tag["K_tax"].isna() & B_tag["K_name"].isna()
    B_tag_nk = B_tag[B_nokey].drop_duplicates().copy()
    B_tag_jn = B_tag[~B_nokey].copy()

    full = A_tag.merge(B_tag_jn, on=["K_tax", "K_name"], how="outer", indicator=True)
    if len(B_tag_nk) > 0:
        B_tag_nk["_merge"] = "right_only"
        full = pd.concat([full, B_tag_nk], ignore_index=True, sort=False)

    for src, alias in [
        (f"{name_col_A}_new", "組織名稱2_x"), (f"{tax_col_A}_new", "統一編號_x"),
        (f"{name_col_B}_old", "組織名稱2_y"), (f"{tax_col_B}_old", "統一編號_y"),
        ("組織名稱2_new", "組織名稱2_x"), ("統一編號_new", "統一編號_x"),
        ("組織名稱2_old", "組織名稱2_y"), ("統一編號_old", "統一編號_y"),
        ("K_name2_new", "K_name2_x"), ("K_name2_old", "K_name2_y"),
    ]:
        if src in full.columns and alias not in full.columns:
            full[alias] = full[src]

    full["K_name2_co"] = full.apply(
        lambda r: coalesce(r.get("K_name2_x"), r.get("K_name2_y"), r.get("K_name2_new"), r.get("K_name2_old")), axis=1)
    full["本期是否出現"] = np.where(full["_merge"].isin(["both", "left_only"]), "本期有", "本期無")

    _prog(5, "偵測異常（08/09 改名 / 錯編）...")
    t08 = full.dropna(subset=["K_tax"]).groupby("K_tax")["K_name2_co"].nunique(dropna=True).reset_index(name="d")
    tax_multi = set(t08.loc[t08["d"] > 1, "K_tax"].astype(str))
    full["08"] = np.where(full["K_tax"].astype(str).isin(tax_multi), "是", "否")

    t09 = full.dropna(subset=["K_name2_co"]).groupby("K_name2_co")["K_tax"].nunique(dropna=True).reset_index(name="d")
    name_multi = set(t09.loc[t09["d"] > 1, "K_name2_co"].astype(str))
    full["09"] = np.where(full["K_name2_co"].astype(str).isin(name_multi), "是", "否")

    prev_tn = pd.DataFrame(columns=["K_tax", "K_name"])
    if P_prev is not None:
        prev_tn = (P_prev.dropna(subset=["K_tax_prev", "K_name_prev"])[["K_tax_prev", "K_name_prev"]]
                   .drop_duplicates().rename(columns={"K_tax_prev": "K_tax", "K_name_prev": "K_name"}))
    cur_tn = full.dropna(subset=["K_tax", "K_name"])[["K_tax", "K_name"]].drop_duplicates()
    tn_all = pd.concat([prev_tn, cur_tn], ignore_index=True)

    t10 = tn_all.dropna(subset=["K_tax"]).groupby("K_tax")["K_name"].nunique(dropna=True).reset_index(name="d")
    tax_m3 = set(t10.loc[t10["d"] > 1, "K_tax"].astype(str))
    full["10"] = np.where(full["K_tax"].astype(str).isin(tax_m3), "是", "否")

    t11 = tn_all.dropna(subset=["K_name"]).groupby("K_name")["K_tax"].nunique(dropna=True).reset_index(name="d")
    name_m3 = set(t11.loc[t11["d"] > 1, "K_name"].astype(str))
    full["11"] = np.where(full["K_name"].astype(str).isin(name_m3), "是", "否")

    rp = full["10"].eq("是"); ri = full["08"].eq("是")
    full["改名_本期vs上期"] = np.where(rp, "是", "否")
    full["改名_本期vs母檔"] = np.where(ri, "是", "否")
    full["改名_三期彙整"]   = np.select([(~rp)&(~ri), rp&ri, rp&(~ri), (~rp)&ri],
        ["否","是","僅本期vs上一期為是","僅本期vs母檔為是"], default="")

    tp = full["11"].eq("是"); ti = full["09"].eq("是")
    full["統編異常_本期vs上期"] = np.where(tp, "是", "否")
    full["統編異常_本期vs母檔"] = np.where(ti, "是", "否")
    full["統編異常_三期彙整"]   = np.select([(~tp)&(~ti), tp&ti, tp&(~ti), (~tp)&ti],
        ["否","是","僅本期vs上一期為是","僅本期vs母檔為是"], default="")

    cols_08 = [c for c in ["K_tax","K_name2_co","組織名稱2_x","統一編號_x","組織名稱2_y","統一編號_y",
        "改名_本期vs上期","改名_本期vs母檔","改名_三期彙整","本期是否出現"] if c in full.columns]
    cols_09 = [c for c in ["K_tax","K_name2_co","組織名稱2_x","統一編號_x","組織名稱2_y","統一編號_y",
        "統編異常_本期vs上期","統編異常_本期vs母檔","統編異常_三期彙整","本期是否出現"] if c in full.columns]
    df_08 = full.loc[full["08"].eq("是"), cols_08].copy()
    df_09 = full.loc[full["09"].eq("是"), cols_09].copy()

    for c in ["08", "09", "10", "11"]:
        if c in full.columns: full.drop(columns=[c], inplace=True)

    _prog(6, "三期分類（持續存在 / 歷史補回 / 本期新增）...")
    full["KEY"] = full["K_tax"].fillna("") + "|" + full["K_name"].fillna("")
    full["上一期有無"]   = np.where(full["KEY"].isin(prev_keys), "上一期有", "上一期無")
    full["初期是否出現"] = np.where(full["KEY"].isin(init_keys), "初期有", "初期無")

    c_keep = (full["本期是否出現"]=="本期有") & (full["上一期有無"]=="上一期有")
    c_back = (full["本期是否出現"]=="本期有") & (full["上一期有無"]=="上一期無") & (full["初期是否出現"]=="初期有")
    c_new  = (full["本期是否出現"]=="本期有") & (full["上一期有無"]=="上一期無") & (full["初期是否出現"]=="初期無")
    full["新增分類"] = np.select([c_keep, c_back, c_new], ["持續存在", "歷史補回", "本期真正新增"], default="")

    for c in ["社創組織資料庫_new", "社創平台網址上架網址_new", "生命力新聞_new"]:
        if c not in full.columns: full[c] = np.nan

    _prog(7, "從 P 檔補值社創 / 生命力新聞欄位...")
    if P_prev is not None and "K_tax_prev" in P_prev.columns and "K_name_prev" in P_prev.columns:
        for bare_col, new_col in [
            ("社創組織資料庫",    "社創組織資料庫_new"),
            ("社創平台網址上架網址", "社創平台網址上架網址_new"),
            ("生命力新聞",        "生命力新聞_new"),
        ]:
            src = _find_prev_col(P_prev, bare_col, new_col)
            if src is None:
                continue
            tmp = (P_prev[["K_tax_prev", "K_name_prev", src]]
                   .drop_duplicates(["K_tax_prev", "K_name_prev"])
                   .rename(columns={"K_tax_prev": "_kt", "K_name_prev": "_kn", src: "_src"}))
            full = full.merge(tmp, left_on=["K_tax", "K_name"], right_on=["_kt", "_kn"], how="left")
            full.drop(columns=["_kt", "_kn"], inplace=True, errors="ignore")
            full[new_col] = full[new_col].combine_first(full["_src"])
            full.drop(columns=["_src"], inplace=True, errors="ignore")

    _prog(8, "產生精簡表...")
    full["組織名稱2_fb"]  = full.apply(lambda r: fb(r,"組織名稱2_x","組織名稱2_y","組織名稱_old"), axis=1)
    full["統一編號_fb"]   = full.apply(lambda r: fb(r,"統一編號_x","統一編號_y"), axis=1)
    full["負責人2_fb"]    = full.apply(lambda r: fb(r,"負責人2_new","負責人2_old"), axis=1)
    full["縣市_fb"]       = full.apply(lambda r: fb(r,"縣市_new","縣市_old"), axis=1)
    full["地址_fb"]       = full.apply(lambda r: fb(r,"地址_new","地址_old"), axis=1)
    full["官方網站_fb"]   = full.apply(lambda r: fb(r,"官方網站_new","官方網站_old"), axis=1)
    full["Facebook_fb"]  = full.apply(lambda r: fb(r,"Facebook_new","Facebook_old"), axis=1)
    full["影音網址_fb"]   = full.apply(lambda r: fb(r,"影音網址_new","影音網址_old"), axis=1)
    full["電子商務_fb"]   = full.apply(lambda r: fb(r,"電子商務_new","電子商務_old"), axis=1)
    full["社創組織資料庫_fb"]       = full.apply(lambda r: fb(r,"社創組織資料庫_new","社創組織資料庫_old"), axis=1)
    full["社創平台網址上架網址_fb"] = full.apply(lambda r: fb(r,"社創平台網址上架網址_new","社創平台網址上架網址_old"), axis=1)
    full["生命力新聞_fb"] = full.apply(lambda r: fb(r,"生命力新聞_new","生命力新聞_old"), axis=1)
    full["K_name2_fb"]   = full.apply(lambda r: fb(r,"K_name2_x","K_name2_y"), axis=1)

    tcols = ["排序2_new","組織名稱2_fb","組織類型_new","負責人2_fb","統一編號_fb",
        "單位聯絡電話(市話)_new","單位聯絡電話-分機_new","單位聯絡手機_new",
        "電子信箱Email_new","單位聯絡人_new","縣市_fb","地址_fb",
        "商品分類一_new","商品分類二_new","商品分類三_new",
        "聯合國永續發展目標-1_new","聯合國永續發展目標-2_new","聯合國永續發展目標-3_new",
        "官方網站_fb","Facebook_fb","影音網址_fb","電子商務_fb",
        "社創組織資料庫_fb","社創平台網址上架網址_fb","生命力新聞_fb",
        "K_tax","K_name","K_name2_fb",
        "改名_本期vs上期","改名_本期vs母檔","改名_三期彙整",
        "統編異常_本期vs上期","統編異常_本期vs母檔","統編異常_三期彙整",
        "上一期有無","初期是否出現","新增分類","是否需要比對_new"]
    for c in tcols:
        if c not in full.columns: full[c] = np.nan
    fdf = full[tcols].copy().rename(columns={
        "組織名稱2_fb":"組織名稱2","統一編號_fb":"統一編號","負責人2_fb":"負責人2",
        "縣市_fb":"縣市","地址_fb":"地址","官方網站_fb":"官方網站","Facebook_fb":"Facebook",
        "影音網址_fb":"影音網址","電子商務_fb":"電子商務","社創組織資料庫_fb":"社創組織資料庫",
        "社創平台網址上架網址_fb":"社創平台網址上架網址","生命力新聞_fb":"生命力新聞","K_name2_fb":"K_name2"})
    fdf = fdf.sort_values("排序2_new", ascending=True, na_position="last").reset_index(drop=True)

    _prog(9, "彙整各工作表...")
    Au  = A_keyed.merge(B_keyed,on=["K_tax","K_name2"],how="left",indicator=True)
    Au  = Au.loc[Au["_merge"].eq("left_only")].drop(columns=["_merge"])
    Bu  = B_keyed.merge(A_keyed,on=["K_tax","K_name2"],how="left",indicator=True)
    Bu  = Bu.loc[Bu["_merge"].eq("left_only")].drop(columns=["_merge"])
    Aut = A.dropna(subset=["K_tax"]).merge(B.dropna(subset=["K_tax"]),on="K_tax",how="left",indicator=True)
    Aut = Aut.loc[Aut["_merge"].eq("left_only")].drop(columns=["_merge"])
    But = B.dropna(subset=["K_tax"]).merge(A.dropna(subset=["K_tax"]),on="K_tax",how="left",indicator=True)
    But = But.loc[But["_merge"].eq("left_only")].drop(columns=["_merge"])

    sheets = {
        "01_多對一原始命中":    many_to_one,
        "02_精簡表(指定欄位)":  fdf,
        "03_一對一結果(B唯一)": final_df,
        "08_同統編異名_疑似改名": df_08,
        "09_同名異統編_疑似錯編": df_09,
        "02_全量結果(含未命中)": full,
    }
    if not Au.empty:  sheets["14_A未命中_雙鍵"] = Au
    if not Bu.empty:  sheets["15_B未命中_雙鍵"] = Bu
    if not Aut.empty: sheets["16_A未命中_統編"] = Aut
    if not But.empty: sheets["17_B未命中_統編"] = But
    if not A_len7.empty: sheets["18_A七碼統編"] = A_len7
    if not B_len7.empty: sheets["19_B七碼統編"] = B_len7
    if not A_pad.empty:  sheets["20_A補零建議"] = A_pad
    if not B_pad.empty:  sheets["21_B補零建議"] = B_pad

    stats = {
        "total":      len(fdf),
        "name_null":  int(fdf["組織名稱2"].isna().sum()),
        "kname_null": int(fdf["K_name"].isna().sum()),
        "rename_08":  len(df_08),
        "taxerr_09":  len(df_09),
        "prev_keys":  len(prev_keys),
        "new_class":  fdf["新增分類"].value_counts(dropna=False).to_dict(),
        "merge_dist": full["_merge"].value_counts().to_dict() if "_merge" in full.columns else {},
    }
    return sheets, stats, fdf

# =========================================================
# Session State 初始化
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# Step 5：格式修正（統編補零 + 手機格式）
# =========================================================
_TAXID_CANDIDATES  = ["統一編號", "統編", "統編碼", "公司統編", "營利事業統一編號"]
_MOBILE_CANDIDATES = ["單位聯絡手機", "聯絡手機", "手機", "手機號碼",
                      "聯絡電話", "電話", "Phone", "Mobile"]

def _is_taxid_col(col_name):
    name = str(col_name).lower()
    if any(c.lower() in name for c in _TAXID_CANDIDATES): return True
    return "統一編號" in col_name or "統編" in col_name

def _is_mobile_col(col_name):
    name = str(col_name).lower()
    if any(c.lower() in name for c in _MOBILE_CANDIDATES): return True
    return any(k in col_name for k in ["手機", "電話", "聯絡"])

def _fix_taxid(x):
    s = str(x or "").strip()
    m = re.match(r"^(-?\d+)\.0+$", s)
    if m: s = m.group(1)
    s = s.replace(" ", "")
    return s.zfill(8) if s.isdigit() and len(s) < 8 else s

def _fix_mobile(x):
    s_raw = str(x or "").strip()
    m = re.match(r"^(-?\d+)\.0+$", s_raw)
    if m: s_raw = m.group(1)
    if s_raw.startswith("+"):
        return "+" + re.sub(r"[\s\-\./]", "", s_raw[1:])
    s = re.sub(r"[\s\-\./]", "", s_raw)
    if s.isdigit() and len(s) == 9 and s[0] == "9": return "0" + s
    if s.isdigit() and len(s) == 10 and s[0] != "0" and s[1] == "9": return "0" + s[1:]
    return s

def run_format_fix(sheets_dict, target_sheets=None):
    """
    sheets_dict : {sheet_name: DataFrame}
    target_sheets : list of sheet names to process；None = all
    回傳 (fixed_dict, log) — fixed_dict 同結構，log 為說明字串清單
    """
    log = []
    fixed = {}
    process = {k: v for k, v in sheets_dict.items()
               if target_sheets is None or k in target_sheets}

    for sname, df in sheets_dict.items():
        if sname not in process:
            fixed[sname] = df
            continue

        df_proc = df.copy()
        tax_cols    = [c for c in df_proc.columns if _is_taxid_col(c)]
        mobile_cols = [c for c in df_proc.columns if _is_mobile_col(c)]

        if not tax_cols and not mobile_cols:
            log.append(f"⚠️ {sname}：未找到統編或手機欄位，略過")
            fixed[sname] = df_proc
            continue

        for col in tax_cols:
            before = df_proc[col].head(3).tolist()
            df_proc[col] = df_proc[col].astype(str).map(_fix_taxid)
            after  = df_proc[col].head(3).tolist()
            log.append(f"✔ {sname} / {col}：{before} → {after}")

        for col in mobile_cols:
            before = df_proc[col].head(3).tolist()
            df_proc[col] = df_proc[col].astype(str).map(_fix_mobile)
            after  = df_proc[col].head(3).tolist()
            log.append(f"✔ {sname} / {col}：{before} → {after}")

        fixed[sname] = df_proc

    return fixed, log

def make_excel_text_fmt(sheets_dict):
    """用 xlsxwriter 輸出，統編/手機欄位強制套用文字格式（防 Excel 吃掉前導零）"""
    import xlsxwriter
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for sname, df in sheets_dict.items():
            df.to_excel(w, index=False, sheet_name=sname[:31])
            ws  = w.sheets[sname[:31]]
            wb  = w.book
            fmt = wb.add_format({"num_format": "@"})
            for col_name in df.columns:
                if _is_taxid_col(col_name) or _is_mobile_col(col_name):
                    idx = df.columns.get_loc(col_name)
                    ws.set_column(idx, idx, 18, fmt)
    return buf.getvalue()

# =========================================================
# UI — 5 個 Tab
# =========================================================
st.title("📊 生命力比對系統")
st.caption("完整工作流程：文章去重 → 文字探勘 → 比對分析 → 文章比對 → 格式修正")

with st.sidebar:
    st.markdown("## 🧭 作業流程（A → E）")
    with st.expander("**A. 名單差異比對**", expanded=False):
        st.markdown("""
**輸入：**
- A = 最新社創名單（已補 組織名稱2 / 負責人2 / 序號2）
- B = 上一期社創名單

**處理：**
以主鍵（建議：統編；若無，採 組織名稱2 + 縣市 + 類型 組合鍵）比對 A vs B。
標記差異：
- **命中**：A、B 都有
- **僅A**：只在 A（新出現）
- **僅B**：只在 B（可能下架 / 異動，列入觀察）

將原始社創欄位 + 既有生命力文章結果合流至最新名單 A。

**輸出：**
- 名單差異摘要（新增 / 保留 / 移除 統計）
- A_整併結果.xlsx（供後續文章比對）
        """)

    with st.expander("**B. 匯入生命力新聞，清洗去重與關鍵字整理**", expanded=False):
        st.markdown("""
**輸入：** 歷史至今所有文章（不同版面可能重複）

**處理：**
- 以 url（或正規化後 url_norm）去重，保留唯一文章
- 產出 / 校正欄位：article_title, url, parsed_date, article_id, keywords
- keywords：jieba TF-IDF / TextRank + 規則式關鍵詞（人名 / 組織 / 地名白名單）

**輸出：**
- articles_deduped.xlsx（唯一文章表）
- （可選）keywords_audit.xlsx（高頻可疑詞審核清單）

👉 對應系統步驟：**Step 1 文章去重** → **Step 2 文字探勘**
        """)

    with st.expander("**C. 依命中狀態決定比對範圍**", expanded=False):
        st.markdown("""
**規則：**
- **命中**（A、B 均有）：只比對新增期間文章（上次比對日 至今）
- **僅A**（本期新出現）：比對全期間文章（回補歷史關聯）

**處理：**
- 對每筆組織依狀態套用時間窗
- 建立比對詞集：正式名稱 + 常見別名 / 縮寫（必要時加 負責人 / 品牌名）
- 與 articles_deduped.keywords 做交集或模糊比對

**輸出：**
- 比對結果_逐筆（含：命中狀態、比對期間、文章清單、比對到的關鍵字）
- 比對結果整理（彙總：每組織命中篇數、最新文章日期）

👉 對應系統步驟：**Step 3 比對分析** → **Step 4 文章比對**
        """)

    with st.expander("**D. 可疑名稱人工審核與社創資格確認**", expanded=False):
        st.markdown("""
**情境 1：無法辨識名稱**（如 林小姐、小明基金會志工）
→ 人工開啟原文頁面核對脈絡（是否指涉該組織 / 是否具官方身分）

**情境 2：社創資格待確認**
→ 至社會創新平台搜尋；必要時查經濟部登錄資料與官網

**輸出：**
人工審核標記（通過 / 排除 / 待查）、審核依據網址、審核日期、審核人
        """)

    with st.expander("**E. 上架與產出**", expanded=False):
        st.markdown("""
**上架範圍：** 命中 與 僅A 且「人工審核通過」者

**對外產出：** 上架文章（附來源、核對日期；必要時加「組織簡介」）

**對內產出：**
- 02_精簡表(指定欄位)（供小編操作）
- 比對結果整理（管理者檢視）
- 疑義表（待查事項追蹤）

👉 對應系統步驟：**Step 5 格式修正** 後匯出
        """)

    st.divider()
    st.markdown("### ✅ 品質檢核清單")
    st.markdown("""
- 欄位對齊：組織名稱2_x / 負責人2_new / 生命力新聞_new 皆存在且有值
- keywords_audit 高頻可疑詞已處理
- 比對結果整理「僅A」確屬新出現（抽查 3–5 筆）
- 命中文章時間窗已正確套用「新增期間」
- 上架清單皆已填 來源 與 核對日期
    """)

    st.divider()
    st.markdown("### 🛠️ 後續可優化")
    st.markdown("""
- **別名庫**：為常見別稱 / 縮寫建立白名單
- **暱稱偵測**：對人名+職稱（小姐 / 老師 / 志工）設停用規則
- **自動開窗**：依上期「最後比對日」自動推算本期「新增期間」
- **審核面板**：疑義表視覺化，支援一鍵標記
    """)

    st.divider()
    st.markdown("### 🔧 欄位命名對照")
    st.markdown("""
| 來源 | 欄位 |
|------|------|
| 社創名單 | 統編, 組織名稱2, 負責人2, 序號2 |
| 主檔合併後 | 組織名稱2_x, 負責人2_new, 生命力新聞_new |
| 文章資料 | article_title, url, parsed_date, article_id, keywords |
    """)

tab_dedup, tab_mining, tab_match, tab_compare, tab_fmt, tab_history = st.tabs([
    "📰 Step 1：文章去重",
    "🔍 Step 2：文字探勘",
    "🔗 Step 4：文章比對",
    "📊 Step 3：比對分析",
    "🔧 Step 5：格式修正",
    "📁 歷史記錄",
])

# ── Step 1：文章去重 ───────────────────────────────────────
with tab_dedup:
    st.subheader("文章去重（依 article_title + url 去除重複）")
    st.markdown("""
    **輸入格式**：Excel 檔，需包含 `article_title` 和 `url` 兩個欄位。
    **輸出**：去重後唯一文章（unique_articles）＋重複清單（duplicates_summary）。
    """)

    dedup_file = st.file_uploader("上傳文章 Excel (.xlsx)", type=["xlsx"], key="dedup_file")
    dedup_sheet = st.text_input("工作表名稱（留空則自動偵測）", value="", key="dedup_sheet",
                                placeholder="links / 工作表1 / 留空自動")

    if st.button("🚀 開始去重", type="primary", disabled=(dedup_file is None),
                 use_container_width=True, key="dedup_run"):
        with st.spinner("讀取並去重中..."):
            try:
                sn = dedup_sheet.strip() if dedup_sheet.strip() else 0
                try:
                    df_in = pd.read_excel(dedup_file, sheet_name=sn)
                except Exception:
                    dedup_file.seek(0)
                    xf = pd.ExcelFile(dedup_file)
                    fallback = "unique_articles" if "unique_articles" in xf.sheet_names else xf.sheet_names[0]
                    dedup_file.seek(0)
                    df_in = pd.read_excel(dedup_file, sheet_name=fallback)

                df_unique, df_dupes, total = run_dedup(df_in)

                st.success(f"去重完成！原始 {total} 筆 → 唯一 {len(df_unique)} 筆，重複組合 {len(df_dupes)} 組")

                m1, m2, m3 = st.columns(3)
                m1.metric("原始筆數", total)
                m2.metric("去重後筆數", len(df_unique))
                m3.metric("移除重複筆數", total - len(df_unique))

                if not df_dupes.empty:
                    st.markdown("**重複組合預覽（前 20 組）**")
                    st.dataframe(df_dupes.head(20), use_container_width=True, hide_index=True)

                # 產生下載 Excel
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    df_unique.to_excel(w, sheet_name="unique_articles", index=False)
                    df_dupes.to_excel(w, sheet_name="duplicates_summary", index=False)
                fname_dedup = f"articles_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                st.download_button(
                    label="⬇️ 下載去重結果 Excel",
                    data=buf.getvalue(),
                    file_name=fname_dedup,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"去重失敗：{e}")

# ── Step 2：文字探勘 ───────────────────────────────────────
with tab_mining:
    st.subheader("文字探勘（網路爬蟲 + jieba TF-IDF / TextRank）")
    st.markdown("""
    **輸入格式**：Excel 檔，需有 `url` 欄位（`article_title` 選填）。
    建議先執行 Step 1 去重後再上傳。
    **輸出欄位**：`article_title`、`parsed_date`、`article_id`、`jieba_keywords`、
    `rule_keywords`、`keywords`、`fetch_error`、`content_chars`、`raw_preview`
    """)

    st.warning(
        "網路爬蟲速度依文章數量和網路狀況而定，1,400 篇約需 3–5 分鐘（爬取）+ 最長 90 分鐘（jieba 分析）。"
        "請勿關閉視窗。建議在本機執行大量資料；Streamlit Cloud 有連線逾時風險。",
        icon="⏱️",
    )

    mining_file   = st.file_uploader("上傳文章 Excel (.xlsx)", type=["xlsx"], key="mining_file")
    mining_sheet  = st.text_input("工作表名稱（留空則自動偵測）", value="", key="mining_sheet",
                                  placeholder="links / unique_articles / 留空自動")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        max_workers = st.number_input("爬蟲平行執行緒數", min_value=1, max_value=20, value=6)
    with col_m2:
        timeout_sec = st.number_input("單篇爬取逾時（秒）", min_value=3, max_value=30, value=5)
    with col_m3:
        topk = st.number_input("jieba 關鍵字數上限（topK）", min_value=5, max_value=50, value=15)
    use_textrank = st.checkbox("同時使用 TextRank（較慢，關鍵字更豐富）", value=True)

    if st.button("🚀 開始文字探勘", type="primary", disabled=(mining_file is None),
                 use_container_width=True, key="mining_run"):

        # 讀取檔案
        try:
            sn = mining_sheet.strip() if mining_sheet.strip() else 0
            try:
                df_in = pd.read_excel(mining_file, sheet_name=sn)
                used_sheet = sn if sn != 0 else "（第一個工作表）"
            except Exception:
                mining_file.seek(0)
                xf = pd.ExcelFile(mining_file)
                fallback = "unique_articles" if "unique_articles" in xf.sheet_names else xf.sheet_names[0]
                mining_file.seek(0)
                df_in = pd.read_excel(mining_file, sheet_name=fallback)
                used_sheet = fallback
            st.info(f"使用工作表：{used_sheet}，共 {len(df_in)} 筆")
        except Exception as e:
            st.error(f"讀取失敗：{e}")
            st.stop()

        # 進度顯示
        progress_bar = st.progress(0, text="準備中...")
        progress_text = st.empty()

        def _mining_progress(pct, msg):
            progress_bar.progress(pct, text=msg)
            phase = "階段 1/2：網路爬取" if pct <= 0.5 else "階段 2/2：jieba 關鍵字分析"
            progress_text.caption(phase)

        try:
            result_df = run_text_mining(
                df_in,
                max_workers=int(max_workers),
                timeout_sec=int(timeout_sec),
                topk=int(topk),
                use_textrank=use_textrank,
                progress_cb=_mining_progress,
            )
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.error(f"文字探勘失敗：{e}")
            st.stop()

        progress_bar.progress(1.0, text="完成！")
        progress_text.empty()
        unknown_ct = int((result_df["parsed_date"] == "unknown").sum())
        err_ct     = int(result_df["fetch_error"].astype(bool).sum())
        st.success(f"文字探勘完成！共 {len(result_df)} 篇，未知日期 {unknown_ct} 篇，爬取失敗 {err_ct} 篇")

        preview_cols = [c for c in ["article_title","parsed_date","article_id","keywords","fetch_error"]
                        if c in result_df.columns]
        st.markdown("**預覽（前 20 筆）**")
        st.dataframe(result_df[preview_cols].head(20), use_container_width=True, hide_index=True)

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            result_df.to_excel(w, sheet_name="links", index=False)
        fname_mining = f"articles_with_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        st.download_button(
            label="⬇️ 下載文字探勘結果 Excel",
            data=buf.getvalue(),
            file_name=fname_mining,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ── Step 4：文章比對 ───────────────────────────────────────
with tab_match:
    st.subheader("文章比對（Step 2 關鍵字 × Step 3 組織名稱）")
    st.markdown("""
    **輸入**：Step 2 文字探勘輸出 ＋ Step 3 比對分析輸出
    **輸出**：更新後的比對結果，新增欄位：
    - `生命力新聞_new_match`：本次新命中的文章 URL
    - `比對到的關鍵字`：命中的關鍵字
    - `比對狀態`：本期新增 / 僅歷史
    - 工作表 `比對結果整理`：每篇文章對應到哪些組織
    """)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Step 2 輸出檔** ✱必填（含 keywords 欄位）")
        match_kw_file = st.file_uploader("上傳文字探勘結果 (.xlsx)", type=["xlsx"], key="match_kw")
        match_kw_sheet = st.text_input("工作表名稱", value="links", key="match_kw_sheet")
    with col_m2:
        st.markdown("**Step 3 輸出檔** ✱必填")
        match_main_file = st.file_uploader("上傳比對分析結果 (.xlsx)", type=["xlsx"], key="match_main")
        match_main_sheet = st.text_input("工作表名稱", value="02_精簡表(指定欄位)", key="match_main_sheet")

    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.markdown("**上一期結果** *(選填，用於合併舊關鍵字)*")
        match_prev_file = st.file_uploader("上傳上一期比對結果 (.xlsx)", type=["xlsx"], key="match_prev")
        match_prev_sheet = st.text_input("工作表名稱", value="02_精簡表(指定欄位)", key="match_prev_sheet")
    with col_m4:
        new_date_str = st.text_input(
            "「新文章」起始日期（YYYYMMDD）",
            value="",
            placeholder="留空表示全部文章都視為新文章",
            help="早於此日期的文章只用於「本期真正新增」和「歷史補回」類別，不用於「持續存在」類別",
        )

    match_btn = st.button(
        "🚀 開始文章比對", type="primary",
        disabled=(match_kw_file is None or match_main_file is None),
        use_container_width=True, key="match_run",
    )

    if match_btn:
        with st.spinner("讀取檔案中..."):
            try:
                kw_sn = match_kw_sheet.strip() or 0
                try:
                    kw_df_in = pd.read_excel(match_kw_file, sheet_name=kw_sn)
                except Exception:
                    match_kw_file.seek(0)
                    xf = pd.ExcelFile(match_kw_file)
                    fb_sheet = "links" if "links" in xf.sheet_names else xf.sheet_names[0]
                    match_kw_file.seek(0)
                    kw_df_in = pd.read_excel(match_kw_file, sheet_name=fb_sheet)

                main_sn = match_main_sheet.strip() or 0
                df_main_in = pd.read_excel(match_main_file, sheet_name=main_sn)

                prev_kw_map = {}
                if match_prev_file:
                    try:
                        prev_sn = match_prev_sheet.strip() or 0
                        df_prev = pd.read_excel(match_prev_file, sheet_name=prev_sn)
                        df_prev.columns = pd.Index(df_prev.columns).map(
                            lambda s: str(s or "").replace("\u3000", " ").strip())
                        name_col = "組織名稱2" if "組織名稱2" in df_prev.columns else None
                        if name_col and "比對到的關鍵字" in df_prev.columns:
                            prev_kw_map = dict(zip(
                                df_prev[name_col].astype(str),
                                df_prev["比對到的關鍵字"].astype(str),
                            ))
                    except Exception as e:
                        st.warning(f"上一期結果讀取失敗，略過：{e}")

                new_date_int = 0
                if new_date_str.strip():
                    try:
                        new_date_int = int(new_date_str.strip())
                    except ValueError:
                        st.warning("日期格式錯誤，應為 YYYYMMDD，已略過日期篩選。")

                st.info(f"關鍵字檔：{len(kw_df_in)} 筆，比對主檔：{len(df_main_in)} 筆")
            except Exception as e:
                st.error(f"讀取失敗：{e}")
                st.stop()

        prog_bar  = st.progress(0, text="準備中...")
        prog_text = st.empty()

        def _match_progress(pct, msg):
            prog_bar.progress(pct, text=msg)
            prog_text.caption(msg)

        try:
            df_main_out, df_report = run_article_match(
                kw_df_in, df_main_in,
                new_date_int=new_date_int,
                prev_kw_map=prev_kw_map,
                progress_cb=_match_progress,
            )
        except Exception as e:
            prog_bar.empty(); prog_text.empty()
            st.error(f"文章比對失敗：{e}")
            st.stop()

        prog_bar.progress(1.0, text="完成！")
        prog_text.empty()

        hit_ct    = int((df_main_out["比對狀態"] == "本期新增").sum())
        hist_ct   = int((df_main_out["比對狀態"] == "僅歷史").sum())
        art_hit   = int((df_report["比對狀態"] == "有").sum())
        st.success(f"比對完成！組織命中：本期新增 {hit_ct} 筆 / 僅歷史 {hist_ct} 筆；文章有命中：{art_hit} / {len(df_report)} 篇")

        st.markdown("**組織比對結果預覽（前 30 筆）**")
        preview_m = [c for c in ["組織名稱2","新增分類","生命力新聞","生命力新聞_new_match","比對到的關鍵字","比對狀態"]
                     if c in df_main_out.columns]
        st.dataframe(df_main_out[preview_m].head(30), use_container_width=True, hide_index=True)

        # 輸出 Excel（保留主檔其他工作表）
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df_main_out.to_excel(w, sheet_name="02_精簡表(指定欄位)", index=False)
            df_report.to_excel(w, sheet_name="比對結果整理", index=False)
        fname_match = f"比對結果_更新後_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        st.download_button(
            label="⬇️ 下載文章比對結果 Excel",
            data=buf.getvalue(),
            file_name=fname_match,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ── Step 3：比對分析 ───────────────────────────────────────
with tab_compare:
    st.subheader("比對分析（A × B × P 三期稽核）")
    st.caption("上傳社創資料庫（A）、生命力新聞（B）、上一期結果（P）、最初期結果（I），執行三期稽核比對分析。")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**A 檔｜社創組織登錄資料庫（本期）** ✱必填")
        file_a = st.file_uploader("上傳 A 檔 (.xlsx)", type=["xlsx"], key="fa")
    with col2:
        st.markdown("**B 檔｜最初期比對結果** ✱必填")
        st.caption("小編維護的生命力新聞組織清單（含統一編號），同時作為「初期是否出現」的判斷基準")
        file_b = st.file_uploader("上傳 B 檔 (.xlsx)", type=["xlsx"], key="fb_")

    col3, _ = st.columns(2)
    with col3:
        st.markdown("**P 檔｜上一期比對結果** *(選填，用於三期稽核)*")
        file_p = st.file_uploader("上傳 P 檔 (.xlsx)", type=["xlsx"], key="fp")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        label = st.text_input("本次比對標籤（如：115年3月）", placeholder="用於歷史記錄")
    with col_opt2:
        sheet_p = st.text_input("P 檔工作表名稱", value="02_精簡表(指定欄位)")

    run_btn = st.button("🚀 開始比對", type="primary",
                        disabled=(file_a is None or file_b is None),
                        use_container_width=True)

    if run_btn:
        with st.spinner("讀取檔案中..."):
            try:
                a_tmp = pd.read_excel(file_a, nrows=1)
                tax_a = find_tax_col_from_cols(a_tmp.columns)
                file_a.seek(0)
                A = pd.read_excel(file_a, dtype={tax_a: "string"} if tax_a else None)

                b_tmp = pd.read_excel(file_b, nrows=1)
                tax_b = find_tax_col_from_cols(b_tmp.columns)
                file_b.seek(0)
                B = pd.read_excel(file_b, dtype={tax_b: "string"} if tax_b else None)

                P_prev = None
                if file_p:
                    try:
                        sn = sheet_p.strip() if sheet_p.strip() else 0
                        P_prev = pd.read_excel(file_p, sheet_name=sn,
                            dtype={"統一編號_final":"string","組織名稱2_final":"string",
                                   "統一編號_x":"string","組織名稱2_x":"string",
                                   "統一編號":"string","組織名稱2":"string"})
                    except Exception as e:
                        st.warning(f"P 檔讀取失敗，以無上期資料執行：{e}")

            except Exception as e:
                st.error(f"讀取檔案錯誤：{e}")
                st.stop()

        prog_bar  = st.progress(0, text="準備中...")
        prog_text = st.empty()

        def _analysis_progress(pct, msg):
            prog_bar.progress(pct, text=msg)
            prog_text.caption(f"步驟 {round(pct * 9)}/9：{msg}")

        try:
            sheets, stats, fdf = run_analysis(A, B, P_prev, progress_cb=_analysis_progress)
        except Exception as e:
            prog_bar.empty()
            prog_text.empty()
            st.error(f"分析失敗：{e}")
            st.stop()

        prog_bar.progress(1.0, text="完成！")
        prog_text.empty()

        excel_bytes = make_excel(sheets)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname = (f"比對結果_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                 if label else f"比對結果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        st.session_state.history.insert(0, {
            "label":  label or "（未命名）",
            "ts":     ts,
            "file_a": file_a.name,
            "file_b": file_b.name,
            "file_p": file_p.name if file_p else "（無）",
            "fname":  fname,
            "stats":  stats,
            "excel":  excel_bytes,
        })

        st.success("✅ 比對完成！")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("總筆數",      stats["total"])
        m2.metric("名稱空值",    stats["name_null"])
        m3.metric("K_name 空值", stats["kname_null"])
        m4.metric("08 疑似改名", stats["rename_08"])
        m5.metric("09 疑似錯編", stats["taxerr_09"])
        m6.metric("上期 KEY 數", stats["prev_keys"])

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**新增分類分布**")
            nc = stats["new_class"]
            st.dataframe(pd.DataFrame({"類別": list(nc.keys()), "筆數": list(nc.values())}),
                         use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**merge 分布**")
            mc = stats["merge_dist"]
            st.dataframe(pd.DataFrame({"類型": list(mc.keys()), "筆數": list(mc.values())}),
                         use_container_width=True, hide_index=True)

        preview_cols = [c for c in ["組織名稱2","統一編號","縣市","新增分類","上一期有無",
                                    "初期是否出現","改名_三期彙整","統編異常_三期彙整"]
                        if c in fdf.columns]
        st.markdown("**🔎 02_精簡表 預覽（前 50 筆）**")
        st.dataframe(fdf[preview_cols].head(50), use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ 下載完整比對結果 Excel（多工作表）",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ── Step 5：格式修正 ───────────────────────────────────────
with tab_fmt:
    st.subheader("格式修正（統編補零 + 手機格式）")
    st.markdown("""
    **輸入**：任一比對結果 Excel（通常是 Step 4 輸出）。
    **功能**：
    - **統一編號**：去除浮點小數（`24883973.0` → `24883973`）、補前導零至 8 碼
    - **手機號碼**：去除分隔符、9 碼補 `0`（`933769550` → `0933769550`）
    - 輸出時套用 Excel 文字格式，防止 Excel 再次吃掉前導零
    """)

    fmt_file = st.file_uploader("上傳 Excel 檔 (.xlsx)", type=["xlsx"], key="fmt_file")

    if fmt_file:
        xf = pd.ExcelFile(fmt_file)
        all_sheets = xf.sheet_names
        fmt_file.seek(0)

        selected_sheets = st.multiselect(
            "選擇要處理的工作表（留空表示全部）",
            options=all_sheets,
            default=[],
            key="fmt_sheets",
        )

        fmt_btn = st.button("🚀 開始格式修正", type="primary",
                            use_container_width=True, key="fmt_run")

        if fmt_btn:
            with st.spinner("讀取所有工作表..."):
                try:
                    fmt_file.seek(0)
                    sheets_dict = pd.read_excel(fmt_file, sheet_name=None)
                except Exception as e:
                    st.error(f"讀取失敗：{e}")
                    st.stop()

            target = selected_sheets if selected_sheets else None

            with st.spinner("修正格式中..."):
                try:
                    fixed_dict, log = run_format_fix(sheets_dict, target_sheets=target)
                    excel_bytes = make_excel_text_fmt(fixed_dict)
                except Exception as e:
                    st.error(f"格式修正失敗：{e}")
                    st.stop()

            st.success(f"格式修正完成！共處理 {len(fixed_dict)} 個工作表")

            with st.expander("修正詳細記錄", expanded=True):
                for line in log:
                    st.write(line)

            fname_fmt = f"比對結果_修正後_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            st.download_button(
                label="⬇️ 下載格式修正後 Excel",
                data=excel_bytes,
                file_name=fname_fmt,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# ── 歷史記錄 ──────────────────────────────────────────────
with tab_history:
    st.subheader("本次工作階段的比對記錄")
    st.caption("⚠️ 歷史記錄僅保留於目前瀏覽器工作階段，重新整理頁面後會清除。請及時下載 Excel。")

    if not st.session_state.history:
        st.info("尚無記錄，執行比對後自動出現。")
    else:
        for i, rec in enumerate(st.session_state.history):
            with st.expander(f"🗂️ {rec['ts']} ── {rec['label']}", expanded=(i == 0)):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"- **A 檔**：{rec['file_a']}")
                    st.markdown(f"- **B 檔**：{rec['file_b']}")
                    st.markdown(f"- **P 檔**：{rec['file_p']}")
                with c2:
                    s = rec["stats"]
                    st.markdown(f"- **總筆數**：{s['total']}")
                    st.markdown(f"- **08 疑似改名**：{s['rename_08']}")
                    st.markdown(f"- **09 疑似錯編**：{s['taxerr_09']}")
                    nc = s.get("new_class", {})
                    st.markdown(f"- **持續存在**：{nc.get('持續存在', 0)} ／ **本期新增**：{nc.get('本期真正新增', 0)}")
                st.download_button(
                    label="⬇️ 下載此次 Excel",
                    data=rec["excel"],
                    file_name=rec["fname"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{i}",
                )
