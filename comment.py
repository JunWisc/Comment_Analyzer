import argparse
import json
import sys
from urllib.parse import urlparse, parse_qs

import requests
from google import genai
from google.genai import types
from bs4 import BeautifulSoup 


# 여기에 Gemini API 키 넣기
GEMINI_API_KEY = " "

# 여기에 YouTube Data API v3 키 넣기
YOUTUBE_API_KEY = " "

GEMINI_MODEL = "gemini-2.5-flash"
MAX_COMMENTS = 10                  # 맨 위 댓글 최대 개수


def normalize_url(url: str) -> str:
    """
    URL 중복 체크용 간단 노멀라이즈: 양끝 공백 제거 + 끝 슬래시 제거.
    """
    return url.strip().rstrip("/")


# 플랫폼 판별

def detect_platform(url: str) -> str:
    """
    URL 호스트를 보고 reddit / youtube / x / instagram / clien / dcinside / fmkorea / unknown 중 하나를 리턴.
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


# Reddit: top-level 댓글만 수집

def fetch_comments_from_reddit(url: str, max_comments: int = MAX_COMMENTS):
    """
    Reddit 쓰레드 URL에서 맨 위(top-level) 댓글만 최대 max_comments 개까지 가져온다.
    - data[1]["data"]["children"] 에 있는 t1 만 사용
    - 각 t1 의 'replies' 필드는 절대 타지 않는다 → 꼬리 댓글은 완전히 무시
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


# YouTube: top-level 댓글 수집

def extract_youtube_video_id(url: str) -> str:
    """
    여러 형태의 YouTube URL에서 videoId만 뽑아낸다.
    지원:
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
    YouTube Data API v3를 사용해서 top-level 댓글만 최대 max_comments 개까지 가져온다.
    - replies는 포함하지 않음 (commentThreads.list는 원래 top-level 기준)
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


# Clien: HTML 파싱으로 댓글 수집

def fetch_comments_from_clien(url: str, max_comments: int = MAX_COMMENTS):
    """
    클리앙 글 URL에서 댓글 텍스트를 최대 max_comments 개까지 가져온다.
    - clien.net -> www.clien.net 로 통일
    - User-Agent 를 실제 브라우저와 비슷하게 설정
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


# DCInside: HTML 파싱으로 댓글 수집

def fetch_comments_from_dcinside(url: str, max_comments: int = MAX_COMMENTS):
    """
    DCInside 게시글 URL에서 댓글 텍스트를 최대 max_comments 개까지 가져온다.
    - http -> https 통일
    - User-Agent 를 실제 브라우저와 비슷하게 설정
    - HTML 구조가 수시로 바뀔 수 있으므로 여러 candidate selector를 시도
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

    # DCInside는 갤러리마다 구조가 조금씩 달 수 있어서 후보 selector를 넓게 잡음
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

    # fallback: class 이름에 "cmt" 또는 "comment" 포함된 div들
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


# FMKorea: HTML 파싱으로 댓글 수집

def fetch_comments_from_fmkorea(url: str, max_comments: int = MAX_COMMENTS):
    """
    에펨코리아(fmkorea.com) 게시글 URL에서 댓글 텍스트를 최대 max_comments 개까지 가져온다.
    - http -> https 통일
    - /best/8520412622 같은 슬러그 URL을 XE 기본 형식
      /index.php?mid=best&document_srl=8520412622 로 변환해서 요청
    - User-Agent / Referer 를 실제 브라우저처럼 설정
    - HTML 구조가 변동 가능하므로 여러 selector를 순차적으로 시도
    """
    # http -> https
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    # /best/8520412622 같은 형식을 index.php?mid=best&document_srl=8520412622 로 변환
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path

    path_parts = path.strip("/").split("/")
    # 예: ["best", "8520412622"]
    if len(path_parts) >= 2 and path_parts[-1].isdigit():
        board = path_parts[-2]
        doc_id = path_parts[-1]
        # 기존 query 유지 + mid, document_srl 세팅
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

    # 에펨이 커스텀 코드로 막는 경우(예: 430)
    if resp.status_code == 430:
        raise RuntimeError("FMKorea returned status 430 (likely bot protection / anti-scraping).")

    # 다른 4xx/5xx 에러는 그대로 예외
    resp.raise_for_status()

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    comments = []
    seen = set()

    # FMKorea는 XE 기반이라 댓글이 xe_content 안에 있는 경우가 많음
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

    # fallback: class 이름에 "cmt" 또는 "comment" 포함된 요소들
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



# 기타 플랫폼 (stub / TODO)

def fetch_comments_from_x(url: str, max_comments: int = MAX_COMMENTS):
    raise NotImplementedError("X/Twitter 댓글 수집은 아직 구현되지 않았습니다. X API를 사용하세요.")


def fetch_comments_from_instagram(url: str, max_comments: int = MAX_COMMENTS):
    raise NotImplementedError("Instagram 댓글 수집은 아직 구현되지 않았습니다. Instagram Graph API를 사용하세요.")


def fetch_comments(url: str, max_comments: int = MAX_COMMENTS):
    """
    URL을 보고 플랫폼을 판단한 뒤, 해당 플랫폼용 fetcher를 호출한다.
    지금은 Reddit / YouTube / Clien / DCInside / FMKorea 만 실제 구현.
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


# Gemini로 댓글 분석

def analyze_comments_with_gemini(comments, model: str = GEMINI_MODEL):
    """
    comments: 문자열 리스트 (각 원소 = 한 댓글)
    Gemini API를 사용해:
      - 댓글별 키워드
      - 댓글별 sentiment + 톤
      - 한국어/영어 설명
      - 전체 요약 / 전체 키워드 / 전체 감성
    을 JSON 형태로 돌려준다.
    """
    if not comments:
        raise ValueError("분석할 댓글이 없습니다.")

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("Please set a valid Gemini API key in GEMINI_API_KEY at the top of the code.")

    truncated_comments = comments[:MAX_COMMENTS]

    comments_payload = [
        {"index": i, "text": c}
        for i, c in enumerate(truncated_comments, start=1)
    ]

    prompt = f"""
당신은 인터넷 댓글을 정밀하게 분석하는 AI 입니다.
아래는 어떤 게시물에 달린 댓글 리스트입니다. 각 댓글에 대해:

1) 주요 키워드 (3~7개 정도, 영어/한국어 섞여도 됨)
2) sentiment:
   - "positive", "negative", "mixed", "neutral" 중 하나
3) tone:
   - "sarcastic", "ironic", "angry", "disappointed", "supportive",
     "humorous", "neutral" 등 몇 단어로 요약
4) sentiment_reason:
   - 왜 그렇게 판단했는지 한국어로 짧게 설명
5) sentiment_reason_en:
   - 4번의 내용을 영어로 다시 요약해서 설명

또한 전체 스레드 수준에서:
- overall_sentiment: "positive" / "negative" / "mixed" / "neutral"
- summary_ko: 전체 논조를 한국어로 요약 (3~4문장)
- summary_en: 전체 논조를 영어로 요약 (3~4문장)
- top_keywords: 전체 스레드에서 자주 등장하거나 중요한 키워드 (5~10개)

입력으로 주어지는 JSON 형식의 comments를 분석해서,
아래 JSON 스키마를 정확히 만족하는 문자열만 출력하세요.

필수 JSON 스키마:
{{
  "per_comment": [
    {{
      "index": 정수,
      "text": 문자열,
      "sentiment": "positive" | "negative" | "mixed" | "neutral",
      "tone": 문자열,
      "sentiment_reason": 문자열,
      "sentiment_reason_en": 문자열,
      "keywords": [ 문자열, ... ]
    }}
  ],
  "overall": {{
    "overall_sentiment": "positive" | "negative" | "mixed" | "neutral",
    "summary_ko": 문자열,
    "summary_en": 문자열,
    "top_keywords": [ 문자열, ... ]
  }}
}}

여기 input comments JSON 이 있습니다:
{json.dumps(comments_payload, ensure_ascii=False)}
"""

    client = genai.Client(api_key=GEMINI_API_KEY)

    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        print("Warning: Failed to parse model response as JSON. Printing raw text.", file=sys.stderr)
        print(response.text)
        raise

    return result


# 감성 카운트 계산

def compute_sentiment_counts(result: dict):
    per_comment = result.get("per_comment", [])
    sentiment_counts = {}
    for item in per_comment:
        sentiment = item.get("sentiment")
        if not sentiment:
            continue
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    total = sum(sentiment_counts.values())
    return sentiment_counts, total


# 결과 출력

def print_analysis(result: dict):
    """
    analyze_comments_with_gemini() 결과(JSON dict)를 콘솔에 보기 좋게 출력.
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

    print("\n=== Per-Comment Analysis ===")
    for item in per_comment:
        idx = item.get("index")
        txt = item.get("text", "")

        sentiment = item.get("sentiment")
        tone = item.get("tone")
        reason_ko = item.get("sentiment_reason")      # 한국어 설명
        reason_en = item.get("sentiment_reason_en")   # 영어 설명
        keywords = item.get("keywords", [])

        print(f"\n[{idx}] {txt}")
        print(f"  - sentiment           : {sentiment}")
        print(f"  - tone                : {tone}")
        print(f"  - sentiment_reason(ko): {reason_ko}")
        print(f"  - sentiment_reason_en : {reason_en}")
        print(f"  - keywords            : {keywords}")


# 집계 갱신

def update_aggregates(aggregates, lang_group: str, result: dict, platform: str):
    """
    aggregates: {"K": {...}, "E": {...}} 구조
    lang_group: "K" 또는 "E"
    """
    counts, total = compute_sentiment_counts(result)
    group = aggregates[lang_group]

    # sentiment 집계
    for k, v in counts.items():
        group["sentiments"][k] = group["sentiments"].get(k, 0) + v

    # 댓글 수 / URL 수
    group["comments"] += total
    group["url_count"] += 1

    # 플랫폼별 URL 수
    group["platforms"][platform] = group["platforms"].get(platform, 0) + 1

    # keyword 집계
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


# 언어별 비교 숫자 요약

def print_group_summary(label: str, data: dict):
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


# Gemini에게 언어 비교 맡기기 (3문장 한/영)

def analyze_language_comparison_with_gemini(aggregates):
    """
    K/E 양쪽에 데이터가 있을 때, Gemini에게 언어별 차이를 요약하게 맡긴다.
    summary_ko, summary_en 을 각각 정확히 3문장으로 생성하도록 요청한다.
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
당신은 두 언어권(한국어 댓글 그룹 K, 영어권 댓글 그룹 E)의 인터넷 댓글 분석 결과를 비교하여, 
연구 목적에 맞는 요약을 작성하는 전문가입니다.

아래 JSON은 각 언어 그룹별 집계된 정보입니다:
{json.dumps(summary_input, ensure_ascii=False)}

K/E 각각에 대해 다음을 고려해 비교·요약하세요:
- 전체 댓글 수, URL 수
- sentiment 분포 (positive, negative, neutral, mixed)
- 주요 키워드와 담론의 차이
- 어떤 감성이 상대적으로 더 강한지
- 수용자들이 무엇에 열광/비판/냉소/지지하는지

출력 형식은 반드시 다음 JSON 객체 **하나**만 이어서 출력해야 합니다.

{{
  "summary_ko": "문장 3개로 구성된 한국어 요약...",
  "summary_en": "Exactly 3 sentences English summary..."
}}

조건:
1. summary_ko:
   - 한국어 문장 **정확히 3개**로 구성합니다.
   - 각 문장은 공손한 평서체로 끝나야 합니다. (예: '~합니다.', '~입니다.')
   - 줄바꿈 없이 한 줄 문자열로 작성합니다.
   - 예시 스타일: "한국어 댓글은 ~~한 경향을 보입니다. 영어권 댓글은 ~~한 반응이 두드러집니다. 전체적으로 두 그룹은 ~~한 점에서 차이를 보입니다."

2. summary_en:
   - 영어 문장 **정확히 3개**로 구성합니다.
   - 줄바꿈 없이 한 줄 문자열로 작성합니다.
   - 예시 스타일: "Korean comments tend to emphasize ____. English comments show stronger ____ responses. Overall, the two language groups differ in how they frame ____."

3. JSON 이외의 다른 텍스트(설명, 마크다운, 주석 등)는 절대 출력하지 마세요.
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
    세션 전체에서 K/E 그룹 감성 분포를 비교해 숫자 요약을 출력하고,
    조건이 맞으면 Gemini에게 한/영 3문장 요약을 맡긴다.
    """
    print("\n================ Language Comparison Summary ================")
    print_group_summary("Korean (K group)", aggregates["K"])
    print_group_summary("English (E group)", aggregates["E"])

    k = aggregates["K"]
    e = aggregates["E"]
    k_total = k["comments"]
    e_total = e["comments"]

    total_urls = k["url_count"] + e["url_count"]

    # 서로 다른 언어 그룹이 모두 있고, 전체 URL 수가 2개 이상일 때만 AI 기반 비교 실행
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


# top keywords 출력

def print_top_keywords_summary(aggregates, top_n: int = 10):
    """
    각 언어 그룹에서 전체 댓글 기준으로 많이 등장한 keyword 상위 N개 출력.
    - Korean 그룹: K
    - English 그룹: E
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


# 인터랙티브 입력

def ask_lang_group():
    """
    첫 분석 때 K/E 선택을 받는다.
    """
    while True:
        choice = input("Select language group: Korean or English (K/E): ").strip().upper()
        if choice in ("K", "E"):
            return choice
        print("Please enter K or E.")


def ask_next_action(aggregates):
    """
    분석 후에 T / K / E 중 무엇을 할지 묻는다.
    현재까지 몇 개의 URL이 K/E 그룹으로 분석됐는지도 함께 보여준다.
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
    한 번의 URL 분석을 처리한다.
    URL이 잘못되었을 경우 에러로 종료하지 않고, 계속 다시 URL을 묻는다.
    같은 URL을 두 번 넣으면 'Already existed url' 메시지를 띄우고 다시 입력받는다.
    """
    lang_label = "Korean (K)" if lang_group == "K" else "English (E)"

    url = initial_url

    while True:
        if not url:
            url = input("여기! Enter URL to analyze: ").strip()

        normalized = normalize_url(url)

        # 중복 URL 체크
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


# 분석 ui

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
        # 첫 분석
        lang_group = ask_lang_group()
        if args.url:
            print(f"Using URL from command line argument: {args.url}")
            process_one(lang_group, aggregates, seen_urls, initial_url=args.url)
        else:
            process_one(lang_group, aggregates, seen_urls)

        total_runs += 1

        # 이후 반복
        while True:
            action = ask_next_action(aggregates)

            if action == "T":
                print_top_keywords_summary(aggregates, top_n=10)

                if total_runs >= 2:
                    print_comparison_summary(aggregates)
                print("\nExiting program.")
                break

            # action 이 K 또는 E 인 경우
            lang_group = action
            process_one(lang_group, aggregates, seen_urls)
            total_runs += 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
