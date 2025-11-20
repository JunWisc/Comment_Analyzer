import argparse
import json
import sys
from urllib.parse import urlparse, parse_qs

from pathlib import Path
import time

import requests
from google import genai
from google.genai import types
from bs4 import BeautifulSoup


# Put your Gemini API key here
GEMINI_API_KEY = " "

# Put your YouTube Data API v3 key here
YOUTUBE_API_KEY = " "

GEMINI_MODEL = "gemini-2.5-flash"
MAX_COMMENTS = 10  # Max number of top-level comments


# Methodology explanation (for methodology section)
METHODOLOGY_EXPLANATION_KO = (
    "본 연구의 감성 분석은 문맥 기반 자연어 처리가 가능한 대규모 언어 모델인 Google Gemini 2.5 Flash를 활용하여 수행되었습니다. "
    "이는 단순히 긍정/부정 단어의 빈도를 세는 기존의 키워드 매칭 방식과 달리, 텍스트 전체의 의미론적 맥락과 화자의 의도를 종합적으로 평가합니다. "
    "모델은 명시적인 어휘뿐만 아니라 반어법(Sarcasm), 은어, 인터넷 용어 등 미묘한 언어적 뉘앙스를 고려하여 댓글을 4가지 범주(긍정, 부정, 중립, 복합)로 분류했습니다.\n\n"
    "긍정 (Positive): 대상에 대한 동의, 칭찬, 지지, 즐거움 또는 건설적인 피드백을 포함하는 경우.\n\n"
    "부정 (Negative): 대상에 대한 반대, 비판, 분노, 실망감을 표출하거나 모욕적인 언어를 포함하는 경우.\n\n"
    "중립 (Neutral): 강한 감정적 색채나 주관적 의견 없이 사실을 진술하거나 단순한 질문 및 관찰을 하는 경우.\n\n"
    "복합 (Mixed): 칭찬과 비판이 공존하거나, 대상에 대해 양가적인 감정을 드러내는 등 상충되는 정서가 한 문장 내에 포함된 경우."
)

METHODOLOGY_EXPLANATION_EN = (
    "The sentiment analysis was conducted using Google's Gemini 2.5 Flash, a large language model (LLM) capable of context-aware natural language processing. "
    "Unlike traditional keyword-counting methods, this approach evaluates the semantic meaning and underlying intent of each comment. "
    "The model classifies sentiments into four categories (Positive, Negative, Neutral, Mixed) by analyzing not only explicit vocabulary but also linguistic nuances such as sarcasm, irony, slang, and internet-specific terminology.\n\n"
    "Positive: Comments expressing agreement, praise, support, enjoyment, or constructive feedback.\n\n"
    "Negative: Comments expressing disagreement, criticism, anger, disappointment, or using derogatory language.\n\n"
    "Neutral: Factual statements, questions, or observations devoid of strong emotional coloring or subjective opinion.\n\n"
    "Mixed: Comments that contain conflicting emotions, such as combining praise with criticism or expressing ambivalence toward the subject."
)


class StdoutLogger:
    """
    Wrapper around sys.stdout that duplicates everything printed to stdout
    into an internal buffer, so we can save the entire session log to a file.
    """

    def __init__(self, original_stream):
        self.original = original_stream
        self.buffer = []

    def write(self, data):
        # Store log
        self.buffer.append(data)
        # Also print to original stdout
        self.original.write(data)

    def flush(self):
        self.original.flush()

    # Some libraries may call these methods
    def isatty(self):
        if hasattr(self.original, "isatty"):
            return self.original.isatty()
        return False

    def fileno(self):
        if hasattr(self.original, "fileno"):
            return self.original.fileno()
        return 1


def get_result_file_path():
    """
    Decide where to save result.txt.

    - Ask the user for a directory path (or Enter for current dir).
    - Create the directory if needed and return <directory>/result.txt.
    """
    while True:
        base = input(
            "Enter directory path to save result.txt (press Enter for current directory): "
        ).strip()
        if not base:
            directory = Path.cwd()
        else:
            directory = Path(base).expanduser()

        try:
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "result.txt"
            return result_path
        except Exception as e:
            print(f"Invalid directory or cannot create it: {e}")
            # Ask again


def normalize_url(url: str) -> str:
    """
    Simple URL normalization for duplicate checking:
    trims whitespace and removes trailing slash.
    """
    return url.strip().rstrip("/")


# Platform detection

def detect_platform(url: str) -> str:
    """
    Detect platform by URL host and return one of:
    reddit / youtube / x / instagram / clien / dcinside / fmkorea / unknown.
    """
    host = urlparse(url).netloc.lower()
    if "reddit.com" in host:
        return "reddit"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "x.com" in host or "twitter.com" in host:
        return "x"
    if "instagram.com" in host:
        return "instagram"
    if "clien.net" in host:
        return "clien"
    if "dcinside.com" in host or "dcinside.co.kr" in host:
        return "dcinside"
    if "fmkorea.com" in host:
        return "fmkorea"
    return "unknown"


# Reddit: fetch only top-level comments

def fetch_comments_from_reddit(url: str, max_comments: int = MAX_COMMENTS):
    """
    Fetch up to max_comments top-level comments from a Reddit thread URL.
    - Uses data[1]["data"]["children"] and only takes items with kind == "t1"
    - Does NOT traverse the 'replies' field (no nested replies).
    """
    api_url = url
    if not api_url.endswith(".json"):
        api_url = api_url.rstrip("/") + ".json"

    headers = {
        "User-Agent": "CommentAnalyzer/0.1 (by u/your_reddit_username)"
    }

    resp = requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError("Unexpected Reddit JSON structure.")

    comments_root = data[1]["data"]["children"]

    top_level_comments = []

    for item in comments_root:
        if not isinstance(item, dict):
            continue

        kind = item.get("kind")
        data_item = item.get("data", {})

        if kind == "t1":
            body = data_item.get("body")
            if body and body not in ("[deleted]", "[removed]"):
                top_level_comments.append(body)
                if len(top_level_comments) >= max_comments:
                    break

    return top_level_comments


# YouTube: fetch top-level comments

def extract_youtube_video_id(url: str) -> str:
    """
    Extract videoId from various YouTube URL formats.
    Supported examples:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if "youtu.be" in host:
        vid = path.lstrip("/")
        vid = vid.split("/")[0]
        if vid:
            return vid

    if "youtube.com" in host:
        if path == "/watch":
            qs = parse_qs(parsed.query)
            vid = qs.get("v", [""])[0]
            if vid:
                return vid

        if path.startswith("/shorts/"):
            parts = path.split("/")
            if len(parts) >= 3 and parts[2]:
                return parts[2]

        if path.startswith("/embed/"):
            parts = path.split("/")
            if len(parts) >= 3 and parts[2]:
                return parts[2]

    raise ValueError("Could not find video ID in YouTube URL.")


def fetch_comments_from_youtube(url: str, max_comments: int = MAX_COMMENTS):
    """
    Use YouTube Data API v3 to fetch up to max_comments top-level comments.
    - Does not include replies (commentThreads.list is top-level based).
    """
    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        raise RuntimeError("Please set a valid YouTube Data API key in YOUTUBE_API_KEY at the top of the code.")

    video_id = extract_youtube_video_id(url)

    comments = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": YOUTUBE_API_KEY,
            "maxResults": 100,
            "order": "relevance",
            "textFormat": "plainText"
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            top = snippet.get("topLevelComment", {})
            top_snip = top.get("snippet", {})
            text = top_snip.get("textOriginal") or top_snip.get("textDisplay")
            if text:
                comments.append(text)
                if len(comments) >= max_comments:
                    break

        if len(comments) >= max_comments:
            break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments


# Clien: fetch comments by HTML parsing

def fetch_comments_from_clien(url: str, max_comments: int = MAX_COMMENTS):
    """
    Fetch up to max_comments comments from a Clien post URL.
    - Normalize clien.net -> www.clien.net
    - Use a browser-like User-Agent
    - Try multiple candidate CSS selectors as HTML structure may change.
    """
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    parsed = urlparse(url)
    if parsed.netloc == "clien.net":
        parsed = parsed._replace(netloc="www.clien.net")
        url = parsed.geturl()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error during Clien request: {e}")

    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    comments = []
    seen = set()

    candidate_selectors = [
        '[data-role="commentContent"]',
        '.comment_content',
        '.comment_view',
        '.reply_content',
    ]

    for sel in candidate_selectors:
        for elem in soup.select(sel):
            text = elem.get_text(separator="\n", strip=True)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            comments.append(text)
            if len(comments) >= max_comments:
                return comments

    if not comments:
        for elem in soup.find_all(class_=lambda v: v and "comment" in v):
            text = elem.get_text(separator="\n", strip=True)
            if not text or text in seen:
                continue
            seen.add(text)
            comments.append(text)
            if len(comments) >= max_comments:
                break

    return comments


# DCInside: fetch comments by HTML parsing

def fetch_comments_from_dcinside(url: str, max_comments: int = MAX_COMMENTS):
    """
    Fetch up to max_comments comments from a DCInside post URL.
    - Normalize http -> https
    - Use a browser-like User-Agent
    - Try multiple candidate CSS selectors as gallery HTML differs by board.
    """
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error during DCInside request: {e}")

    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    comments = []
    seen = set()

    # DCInside gallery structure may vary, so we try multiple selectors.
    candidate_selectors = [
        "#comment_box .reply",
        "#comment_box .usertxt",
        ".comment_box .reply",
        ".comment_box .usertxt",
        ".comment_box .comment",
        ".ub-content",
        ".comment_cont",
        ".reply",
    ]

    for sel in candidate_selectors:
        for elem in soup.select(sel):
            text = elem.get_text(separator="\n", strip=True)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            comments.append(text)
            if len(comments) >= max_comments:
                return comments

    # Fallback: div elements with class name including "cmt" or "comment"
    if not comments:
        for elem in soup.find_all("div", class_=lambda v: v and ("cmt" in v or "comment" in v)):
            text = elem.get_text(separator="\n", strip=True)
            if not text or text in seen:
                continue
            seen.add(text)
            comments.append(text)
            if len(comments) >= max_comments:
                break

    return comments


# FMKorea: fetch comments by HTML parsing

def fetch_comments_from_fmkorea(url: str, max_comments: int = MAX_COMMENTS):
    """
    Fetch up to max_comments comments from an FMKorea (fmkorea.com) post URL.
    - Normalize http -> https
    - Convert slug URL like /best/8520412622 to XE default:
      /index.php?mid=best&document_srl=8520412622
    - Use browser-like User-Agent and Referer
    - Try multiple candidate CSS selectors as HTML structure may change.
    """
    # http -> https
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    # Convert /best/8520412622 to /index.php?mid=best&document_srl=8520412622
    parsed = urlparse(url)
    path = parsed.path

    path_parts = path.strip("/").split("/")
    # Example: ["best", "8520412622"]
    if len(path_parts) >= 2 and path_parts[-1].isdigit():
        board = path_parts[-2]
        doc_id = path_parts[-1]
        # Keep existing query, add mid and document_srl
        base_query = parse_qs(parsed.query)
        base_query["mid"] = [board]
        base_query["document_srl"] = [doc_id]

        # dict -> query string
        query_items = []
        for k, vs in base_query.items():
            for v in vs:
                query_items.append(f"{k}={v}")
        new_query = "&".join(query_items)

        parsed = parsed._replace(path="/index.php", query=new_query)
        url = parsed.geturl()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Referer": "https://www.fmkorea.com/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error during FMKorea request: {e}")

    # FMKorea may use custom status (e.g., 430) for bot/anti-scraping
    if resp.status_code == 430:
        raise RuntimeError("FMKorea returned status 430 (likely bot protection / anti-scraping).")

    # Other 4xx/5xx errors
    resp.raise_for_status()

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    comments = []
    seen = set()

    # FMKorea is XE-based; comments often live inside xe_content blocks
    candidate_selectors = [
        "#comment .xe_content",
        "#comment .xe_content p",
        "#comment .comment-content",
        ".fdb_lst_wrp .xe_content",
        ".fdb_lst_wrp .comment-content",
        ".comment .xe_content",
    ]

    for sel in candidate_selectors:
        for elem in soup.select(sel):
            text = elem.get_text(separator="\n", strip=True)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            comments.append(text)
            if len(comments) >= max_comments:
                return comments

    # Fallback: elements whose class name includes "cmt" or "comment"
    if not comments:
        for elem in soup.find_all(class_=lambda v: v and ("cmt" in v or "comment" in v)):
            text = elem.get_text(separator="\n", strip=True)
            if not text or text in seen:
                continue
            seen.add(text)
            comments.append(text)
            if len(comments) >= max_comments:
                break

    return comments


# Other platforms (stubs / TODO)

def fetch_comments_from_x(url: str, max_comments: int = MAX_COMMENTS):
    raise NotImplementedError("X/Twitter comment collection is not implemented yet. Please use X API directly.")


def fetch_comments_from_instagram(url: str, max_comments: int = MAX_COMMENTS):
    raise NotImplementedError("Instagram comment collection is not implemented yet. Please use Instagram Graph API.")


def fetch_comments(url: str, max_comments: int = MAX_COMMENTS):
    """
    Detect platform from URL and call the corresponding fetcher.
    Implemented platforms:
      - Reddit / YouTube / Clien / DCInside / FMKorea
    """
    platform = detect_platform(url)
    if platform == "reddit":
        return fetch_comments_from_reddit(url, max_comments=max_comments)
    elif platform == "youtube":
        return fetch_comments_from_youtube(url, max_comments=max_comments)
    elif platform == "clien":
        return fetch_comments_from_clien(url, max_comments=max_comments)
    elif platform == "dcinside":
        return fetch_comments_from_dcinside(url, max_comments=max_comments)
    elif platform == "fmkorea":
        return fetch_comments_from_fmkorea(url, max_comments=max_comments)
    elif platform == "x":
        return fetch_comments_from_x(url, max_comments=max_comments)
    elif platform == "instagram":
        return fetch_comments_from_instagram(url, max_comments=max_comments)
    else:
        raise ValueError(f"Unsupported or unrecognized platform for URL: {url}")


# Analyze comments with Gemini

def analyze_comments_with_gemini(comments, model: str = GEMINI_MODEL):
    """
    Analyze a list of comment texts with Gemini.
    For each comment, the model should produce:
      - keywords
      - sentiment + tone
      - explanation in Korean and English
    For the whole thread, the model should produce:
      - overall sentiment
      - Korean and English summaries
      - top keywords
    Returns the parsed JSON dictionary.
    """
    if not comments:
        raise ValueError("There are no comments to analyze.")

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("Please set a valid Gemini API key in GEMINI_API_KEY at the top of the code.")

    truncated_comments = comments[:MAX_COMMENTS]

    comments_payload = [
        {"index": i, "text": c}
        for i, c in enumerate(truncated_comments, start=1)
    ]

    prompt = f"""
You are an AI specialized in fine-grained analysis of internet comments.
Below is a list of comments posted under a specific content (e.g., a video, article, or forum thread).

For each comment, you must provide:

1) keywords:
   - 3–7 representative keywords (they may be in English or Korean, or mixed)
2) sentiment:
   - exactly one of: "positive", "negative", "mixed", "neutral"
3) tone:
   - a short phrase describing tone, e.g.:
     "sarcastic", "ironic", "angry", "disappointed", "supportive",
     "humorous", "neutral", etc.
4) sentiment_reason:
   - a short explanation in **Korean** describing why you assigned that sentiment
5) sentiment_reason_en:
   - a short explanation in **English** summarizing the same reasoning as (4)

At the overall thread level, you must also provide:

- overall_sentiment:
  - exactly one of: "positive", "negative", "mixed", "neutral"
- summary_ko:
  - a Korean summary of the overall discussion and tone (about 3–4 sentences)
- summary_en:
  - an English summary of the overall discussion and tone (about 3–4 sentences)
- top_keywords:
  - 5–10 important or frequently appearing keywords across all comments

You are given the input comments as a JSON array.
Analyze them and output **only** one JSON object as a string, exactly following the schema below.

Required JSON schema:

{{
  "per_comment": [
    {{
      "index": integer,
      "text": string,
      "sentiment": "positive" | "negative" | "mixed" | "neutral",
      "tone": string,
      "sentiment_reason": string,
      "sentiment_reason_en": string,
      "keywords": [ string, ... ]
    }}
  ],
  "overall": {{
    "overall_sentiment": "positive" | "negative" | "mixed" | "neutral",
    "summary_ko": string,
    "summary_en": string,
    "top_keywords": [ string, ... ]
  }}
}}

Input comments JSON:
{json.dumps(comments_payload, ensure_ascii=False)}
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(response_mime_type="application/json")

    # Retry wrapper for transient 503/UNAVAILABLE errors
    last_err = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            # Try JSON parsing
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                print(
                    "Warning: Failed to parse model response as JSON. Printing raw text.",
                    file=sys.stderr,
                )
                print(response.text, file=sys.stderr)
                # Do not retry on parse error; treat as fatal
                raise

        except Exception as e:
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg:
                print(f"[Warning] Gemini overloaded, retrying... ({attempt+1}/3)")
                last_err = e
                # Wait 3s, 6s, 9s
                time.sleep(3 * (attempt + 1))
                continue
            # Other errors are raised immediately
            raise

    # After 3 attempts, fail
    raise RuntimeError(f"Gemini API still unavailable after retries: {last_err}")


# Sentiment counts

def compute_sentiment_counts(result: dict):
    """
    Count how many comments fall into each sentiment label.
    Returns (sentiment_counts_dict, total_count).
    """
    per_comment = result.get("per_comment", [])
    sentiment_counts = {}
    for item in per_comment:
        sentiment = item.get("sentiment")
        if not sentiment:
            continue
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    total = sum(sentiment_counts.values())
    return sentiment_counts, total


# Print detailed analysis result

def print_analysis(result: dict):
    """
    Pretty-print the result dictionary returned by analyze_comments_with_gemini().
    """
    per_comment = result.get("per_comment", [])
    overall = result.get("overall", {})

    sentiment_counts, total = compute_sentiment_counts(result)

    print("\n========= Sentiment Summary ========")
    if total == 0:
        print("No valid sentiment information.")
    else:
        preferred_order = ["positive", "negative", "neutral", "mixed"]
        printed = set()

        for key in preferred_order:
            if key in sentiment_counts:
                count = sentiment_counts[key]
                pct = (count / total) * 100.0
                print(f"{key:8} : {count:4d} ({pct:5.1f}%)")
                printed.add(key)

        for key, count in sentiment_counts.items():
            if key in printed:
                continue
            pct = (count / total) * 100.0
            print(f"{key:8} : {count:4d} ({pct:5.1f}%)")

    print("\n=== Overall Summary ===")
    print(f"- overall_sentiment: {overall.get('overall_sentiment')}")
    print(f"- summary_ko      : {overall.get('summary_ko')}")
    print(f"- summary_en      : {overall.get('summary_en')}")
    print(f"- top_keywords    : {overall.get('top_keywords')}")

    # Methodology explanation in both Korean and English (for thesis/method section)
    print("\n=== Sentiment Classification Methodology ===")
    print(f"- methodology_ko : {METHODOLOGY_EXPLANATION_KO}")
    print(f"- methodology_en : {METHODOLOGY_EXPLANATION_EN}")

    print("\n=== Per-Comment Analysis ===")
    for item in per_comment:
        idx = item.get("index")
        txt = item.get("text", "")

        sentiment = item.get("sentiment")
        tone = item.get("tone")
        reason_ko = item.get("sentiment_reason")      # Korean explanation
        reason_en = item.get("sentiment_reason_en")   # English explanation
        keywords = item.get("keywords", [])

        print(f"\n[{idx}] {txt}")
        print(f"  - sentiment           : {sentiment}")
        print(f"  - tone                : {tone}")
        print(f"  - sentiment_reason(ko): {reason_ko}")
        print(f"  - sentiment_reason_en : {reason_en}")
        print(f"  - keywords            : {keywords}")


# Update aggregated stats for K/E groups

def update_aggregates(aggregates, lang_group: str, result: dict, platform: str):
    """
    Update aggregate statistics for 'K' or 'E' group
    using the analysis result from a single URL.
    """
    counts, total = compute_sentiment_counts(result)
    group = aggregates[lang_group]

    # Aggregate sentiment counts
    for k, v in counts.items():
        group["sentiments"][k] = group["sentiments"].get(k, 0) + v

    # Aggregate total comments and URLs
    group["comments"] += total
    group["url_count"] += 1

    # Aggregate URL count per platform
    group["platforms"][platform] = group["platforms"].get(platform, 0) + 1

    # Aggregate keyword counts
    per_comment = result.get("per_comment", [])
    for item in per_comment:
        kws = item.get("keywords", []) or []
        for kw in kws:
            if not kw:
                continue
            kw_str = str(kw).strip()
            if not kw_str:
                continue
            group["keywords"][kw_str] = group["keywords"].get(kw_str, 0) + 1


# Print numeric summary for each language group

def print_group_summary(label: str, data: dict):
    """
    Print numeric summary for one language group.
    """
    sentiments = data["sentiments"]
    total = data["comments"]
    url_count = data["url_count"]

    print(f"\n[{label}]")
    print(f"- # of URLs: {url_count}")
    print(f"- Total # of comments: {total}")
    if total == 0:
        print("  (no comment data)")
        return

    for key in ["positive", "negative", "neutral", "mixed"]:
        count = sentiments.get(key, 0)
        pct = (count / total) * 100.0 if total > 0 else 0.0
        print(f"  {key:8} : {count:4d} ({pct:5.1f}%)")


# Ask Gemini for cross-language comparison (3 sentences in KO/EN)

def analyze_language_comparison_with_gemini(aggregates):
    """
    When both K and E groups have data, ask Gemini to summarize
    cross-language differences using aggregate statistics.
    The model must produce summary_ko and summary_en, each with exactly 3 sentences.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n[Warning] Gemini API key is not set. Skipping AI-based comparison.")
        return None

    k = aggregates["K"]
    e = aggregates["E"]

    summary_input = {
        "K": {
            "sentiments": k["sentiments"],
            "comments": k["comments"],
            "url_count": k["url_count"],
            "top_keywords": sorted(
                k["keywords"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:15],
        },
        "E": {
            "sentiments": e["sentiments"],
            "comments": e["comments"],
            "url_count": e["url_count"],
            "top_keywords": sorted(
                e["keywords"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:15],
        },
    }

    prompt = f"""
You are an expert who compares comment analysis results between two language groups:
- K: Korean-language comment group
- E: English-language comment group

Below is aggregated JSON data for each language group:
{json.dumps(summary_input, ensure_ascii=False)}

Using this information, compare and summarize how the two groups differ in:

- total number of comments and URLs
- sentiment distributions (positive, negative, neutral, mixed)
- major keywords and topics
- which emotions are relatively stronger in each group
- what each audience tends to praise, criticize, joke about, or support

Your output must be **exactly one JSON object** with the following structure:

{{
  "summary_ko": "Korean summary with exactly 3 sentences...",
  "summary_en": "English summary with exactly 3 sentences..."
}}

Constraints:

1. summary_ko:
   - Must consist of **exactly 3 sentences in Korean**.
   - Each sentence must end with polite declarative endings, such as "~합니다." or "~입니다.".
   - Write it as a single-line string (no line breaks).
   - Example style: "한국어 댓글은 ~~한 경향을 보입니다. 영어권 댓글은 ~~한 반응이 두드러집니다. 전체적으로 두 그룹은 ~~한 점에서 차이를 보입니다."

2. summary_en:
   - Must consist of **exactly 3 sentences in English**.
   - Write it as a single-line string (no line breaks).
   - Example style: "Korean comments tend to emphasize ____. English comments show stronger ____ responses. Overall, the two language groups differ in how they frame ____."

3. Do NOT output any text other than this single JSON object.
   - No explanations, no markdown, no comments, no additional text.
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(response_mime_type="application/json")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        print("\n[Warning] Failed to parse AI comparison response as JSON. Raw response:")
        print(response.text)
        return None

    return result


def print_comparison_summary(aggregates):
    """
    Print numeric summaries for K/E groups and,
    if there is enough data, call Gemini to produce 3-sentence KO/EN comparison.
    """
    print("\n================ Language Comparison Summary ================")
    print_group_summary("Korean (K group)", aggregates["K"])
    print_group_summary("English (E group)", aggregates["E"])

    k = aggregates["K"]
    e = aggregates["E"]
    k_total = k["comments"]
    e_total = e["comments"]

    total_urls = k["url_count"] + e["url_count"]

    # Require at least 1 URL per group and at least 2 URLs in total
    if total_urls >= 2 and k["url_count"] >= 1 and e["url_count"] >= 1 and k_total > 0 and e_total > 0:
        comp = analyze_language_comparison_with_gemini(aggregates)
        if comp:
            summary_ko = comp.get("summary_ko")
            summary_en = comp.get("summary_en")

            print("\n=== AI-based Comparison (Gemini) ===")
            if summary_ko:
                print("\n[KO]")
                print(summary_ko)
            if summary_en:
                print("\n[EN]")
                print(summary_en)
        else:
            print("\n[Info] Failed to get AI-based comparison; only numeric summary shown above.")
    else:
        print("\n[Info] Not enough cross-language data for AI-based comparison (need at least 2 URLs with both K and E groups).")


# Top keywords summary

def print_top_keywords_summary(aggregates, top_n: int = 10):
    """
    Print top-N keywords for each language group (K and E),
    based on aggregated comment keyword frequencies.
    """
    print("\n================ Top Keywords Summary ================")

    any_data = False

    for code, label in [("K", "Korean (K group)"), ("E", "English (E group)")]:
        group = aggregates[code]
        if group["comments"] == 0:
            continue

        any_data = True
        kw_counts = group.get("keywords", {})
        if not kw_counts:
            print(f"\n[{label}] No keyword data.")
            continue

        sorted_kws = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n[{label}] Top {len(sorted_kws)} keywords:")
        for kw, cnt in sorted_kws:
            print(f"  {kw}: {cnt}")

    if not any_data:
        print("\nNo comment data available for any language group.")


# Interactive input helpers

def ask_lang_group():
    """
    Ask user to choose the first language group: K or E.
    """
    while True:
        choice = input("Select language group: Korean or English (K/E): ").strip().upper()
        if choice in ("K", "E"):
            return choice
        print("Please enter K or E.")


def ask_next_action(aggregates):
    """
    After each analysis, ask user what to do next:
      - T: terminate
      - K: analyze a new URL and count it as Korean-group data
      - E: analyze a new URL and count it as English-group data
    Also shows how many URLs have been analyzed per group.
    """

    def format_platforms(group):
        platforms = group.get("platforms", {})
        if not platforms:
            return "no URLs yet"
        parts = [f"{name} {count}" for name, count in platforms.items()]
        return ", ".join(parts)

    while True:
        k_group = aggregates["K"]
        e_group = aggregates["E"]
        k_urls = k_group["url_count"]
        e_urls = e_group["url_count"]
        k_detail = format_platforms(k_group)
        e_detail = format_platforms(e_group)

        print("\nCurrently analyzed URLs:")
        print(f"  - Korean (K): {k_urls} ({k_detail})")
        print(f"  - English (E): {e_urls} ({e_detail})")

        choice = input(
            "To terminate, enter T. To analyze a new URL, enter K (Korean) or E (English): "
        ).strip().upper()
        if choice in ("T", "K", "E"):
            return choice
        print("Please enter T, K, or E.")


def process_one(lang_group: str, aggregates, seen_urls, initial_url: str = None):
    """
    Process one URL analysis run.
    - Detect platform
    - Fetch comments
    - Analyze with Gemini
    - Print results
    - Update aggregates

    If the URL is invalid or fetching fails, it will ask for a new URL
    instead of crashing.
    It also prevents analyzing exactly the same URL more than once
    within a single run.
    """
    lang_label = "Korean (K)" if lang_group == "K" else "English (E)"

    url = initial_url

    while True:
        if not url:
            url = input("여기! Enter URL to analyze: ").strip()

        normalized = normalize_url(url)

        # Duplicate URL check for current run
        if normalized in seen_urls:
            print("Already existed url")
            url = None
            continue

        try:
            print(f"\n[1/3] ({lang_label}) Detecting platform and fetching comments... ({url})")
            platform = detect_platform(url)
            comments = fetch_comments(url, max_comments=MAX_COMMENTS)
            print(f" - Number of collected top-level comments: {len(comments)}")

            print("[2/3] Analyzing comments with Gemini...")
            result = analyze_comments_with_gemini(comments)

            print("[3/3] Printing results:")
            print_analysis(result)

            update_aggregates(aggregates, lang_group, result, platform)
            seen_urls.add(normalized)
            return

        except Exception as e:
            print(f"\nError while processing URL: {e}", file=sys.stderr)
            print("Please enter the URL again.")
            url = None


def _filter_prompt_lines(log_text: str) -> str:
    """
    Remove the directory prompt line from the saved log text,
    so it does not appear in result.txt.
    """
    lines = log_text.splitlines(keepends=True)
    filtered = []
    for line in lines:
        if line.startswith("Enter directory path to save result.txt"):
            continue
        filtered.append(line)
    return "".join(filtered)


# Main interactive loop

def main():
    parser = argparse.ArgumentParser(
        description="URL-based comment collection + Gemini sentiment/keyword analysis (K/E group comparison)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="First URL to analyze (optional)"
    )
    args = parser.parse_args()

    # Set up stdout logger so we can save the entire console log at the end
    original_stdout = sys.stdout
    logger = StdoutLogger(original_stdout)
    sys.stdout = logger

    aggregates = {
        "K": {
            "sentiments": {},
            "comments": 0,
            "url_count": 0,
            "platforms": {},
            "keywords": {},
        },
        "E": {
            "sentiments": {},
            "comments": 0,
            "url_count": 0,
            "platforms": {},
            "keywords": {},
        },
    }
    total_runs = 0
    seen_urls = set()

    try:
        # First analysis
        lang_group = ask_lang_group()
        if args.url:
            print(f"Using URL from command line argument: {args.url}")
            process_one(lang_group, aggregates, seen_urls, initial_url=args.url)
        else:
            process_one(lang_group, aggregates, seen_urls)

        total_runs += 1

        # Subsequent analyses
        while True:
            action = ask_next_action(aggregates)

            if action == "T":
                # Print top keywords across all runs
                print_top_keywords_summary(aggregates, top_n=10)

                # If we have at least 2 runs, try cross-language comparison
                if total_runs >= 2:
                    print_comparison_summary(aggregates)

                print("\nExiting program.\n")

                # Save all logs to result.txt ONLY when user chose T
                try:
                    result_path = get_result_file_path()
                    log_text = "".join(logger.buffer)
                    log_text = _filter_prompt_lines(log_text)

                    if not log_text.endswith("\n"):
                        log_text += "\n"

                    # Check whether this is the first time writing to this file
                    first_time = (not result_path.exists()) or (result_path.stat().st_size == 0)

                    with result_path.open("a", encoding="utf-8") as f:
                        if first_time:
                            # 1st line: save path + one blank line
                            f.write(f"Saved log to: {result_path}\n\n")
                            # Then methodology explanations (KO + EN)
                            f.write(METHODOLOGY_EXPLANATION_KO + "\n\n")
                            f.write(METHODOLOGY_EXPLANATION_EN + "\n\n")
                        # After that, append this session log
                        f.write(log_text)

                    # Notify user on console (this is not captured into the saved log snapshot)
                    original_stdout.write(f"\n[Info] Saved log to: {result_path}\n")
                except Exception as save_err:
                    original_stdout.write(f"\n[Warning] Failed to save log file: {save_err}\n")

                break

            # If action is K or E, run another analysis under that language group
            lang_group = action
            process_one(lang_group, aggregates, seen_urls)
            total_runs += 1

    except Exception as e:
        # On any error / forced stop, do NOT save logs automatically
        sys.stdout = original_stdout
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore stdout no matter what
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
