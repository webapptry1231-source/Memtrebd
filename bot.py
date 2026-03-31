#!/usr/bin/env python3
"""
Meme Coin Trend Analyzer Bot - Fully Asynchronous with FastAPI & Gemini LLM
Optimized for Railway deployment with real-time trend detection.
"""

import os
import sys
import asyncio
import json
import logging
import traceback
import re
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from io import BytesIO

import aiohttp
import aiosqlite
import pytz
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# -------------------------- Configuration --------------------------
# API Keys & Endpoints
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.grok.ai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MemeCoinBot/1.0")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/search"

# LLM Model Names (configurable)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")

# Defaults
PUMP_SCORE_THRESHOLD = int(os.getenv("PUMP_SCORE_THRESHOLD", "75"))
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "4"))
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ERROR_CHAT_ID = os.getenv("TELEGRAM_ERROR_CHAT_ID", DEFAULT_CHAT_ID)
DB_PATH = "trends.db"

# Daily summary settings
DAILY_SUMMARY_HOUR = int(os.getenv("DAILY_SUMMARY_HOUR", "8"))  # 8 AM
DAILY_SUMMARY_MINUTE = int(os.getenv("DAILY_SUMMARY_MINUTE", "0"))
DAILY_SUMMARY_TIMEZONE = os.getenv("DAILY_SUMMARY_TIMEZONE", "Asia/Kolkata")  # IST

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -------------------------- Database Layer (Async) --------------------------
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                virality_score REAL,
                memeability_score REAL,
                novelty_score REAL,
                crypto_momentum_score REAL,
                pump_score REAL,
                generated_name TEXT,
                generated_ticker TEXT,
                generated_desc TEXT,
                logo_prompt TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_alerted TIMESTAMP,
                user_rating INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pump_score REAL,
                FOREIGN KEY(topic_id) REFERENCES topics(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                chat_id INTEGER PRIMARY KEY,
                threshold INTEGER DEFAULT 75
            )
        """)
        # Table for mention counts – now populated
        await db.execute("""
            CREATE TABLE IF NOT EXISTS topic_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                source TEXT,
                count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def get_user_threshold(chat_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT threshold FROM user_settings WHERE chat_id = ?", (chat_id,))
        row = await cursor.fetchone()
        return row[0] if row else PUMP_SCORE_THRESHOLD

async def set_user_threshold(chat_id: int, threshold: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO user_settings (chat_id, threshold) VALUES (?, ?)",
            (chat_id, threshold)
        )
        await db.commit()

async def save_topic(topic: str, scores: Dict, generated: Optional[Dict] = None, alert: bool = False):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            INSERT INTO topics (
                topic, virality_score, memeability_score, novelty_score,
                crypto_momentum_score, pump_score, generated_name, generated_ticker,
                generated_desc, logo_prompt, created_at, last_alerted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            topic,
            scores.get("virality", 0),
            scores.get("memeability", 0),
            scores.get("novelty", 0),
            scores.get("crypto_momentum", 0),
            scores.get("pump_score", 0),
            generated.get("name") if generated else None,
            generated.get("ticker") if generated else None,
            generated.get("description") if generated else None,
            generated.get("logo_prompt") if generated else None,
            datetime.now(),
            datetime.now() if alert else None,
        ))
        topic_id = cursor.lastrowid
        if alert:
            await db.execute(
                "INSERT INTO alerts (topic_id, pump_score) VALUES (?, ?)",
                (topic_id, scores["pump_score"])
            )
        await db.commit()
        return topic_id

async def get_recent_alert_score(topic: str, hours: int = COOLDOWN_HOURS) -> Optional[float]:
    cutoff = datetime.now() - timedelta(hours=hours)
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT pump_score FROM alerts
            WHERE topic_id IN (SELECT id FROM topics WHERE topic = ?)
            AND sent_at > ?
            ORDER BY sent_at DESC LIMIT 1
        """, (topic, cutoff))
        row = await cursor.fetchone()
        return row[0] if row else None

async def save_mentions(topic: str, source: str, count: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO topic_mentions (topic, source, count) VALUES (?, ?, ?)",
            (topic, source, count)
        )
        await db.commit()

async def get_mention_counts(topic: str, source: str, minutes: int) -> int:
    cutoff = datetime.now() - timedelta(minutes=minutes)
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT SUM(count) FROM topic_mentions
            WHERE topic = ? AND source = ? AND timestamp > ?
        """, (topic, source, cutoff))
        row = await cursor.fetchone()
        return row[0] if row[0] else 0

async def get_latest_scores(limit=50) -> List[Tuple]:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT topic, pump_score, created_at, generated_name, generated_ticker
            FROM topics
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return await cursor.fetchall()

async def get_topics_last_24h(limit=20) -> List[Tuple]:
    """Get topics from last 24 hours for daily summary."""
    cutoff = datetime.now() - timedelta(hours=24)
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT topic, pump_score, created_at, generated_name, generated_ticker
            FROM topics
            WHERE created_at > ?
            ORDER BY pump_score DESC
            LIMIT ?
        """, (cutoff, limit))
        return await cursor.fetchall()

# -------------------------- Helper Functions --------------------------
def normalize_topic(topic: str) -> str:
    cleaned = re.sub(r'[^a-z0-9\s]', '', topic.lower())
    return cleaned.strip()

def extract_json_from_response(response: str) -> Optional[Dict]:
    start = response.find('{')
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(response)):
        if response[i] == '{':
            depth += 1
        elif response[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    json_str = response[start:end+1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

# -------------------------- Async API Helpers --------------------------
class RateLimiter:
    def __init__(self, rate_limit: int, period: float):
        self.rate_limit = rate_limit
        self.period = period
        self.tokens = rate_limit
        self.last_refill = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            while self.tokens <= 0:
                now = asyncio.get_event_loop().time()
                elapsed = now - self.last_refill
                if elapsed >= self.period:
                    self.tokens = self.rate_limit
                    self.last_refill = now
                else:
                    await asyncio.sleep(self.period - elapsed)
            self.tokens -= 1

async def fetch_json(session: aiohttp.ClientSession, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, json_data: Optional[Dict] = None, timeout: int = 30) -> Optional[Dict]:
    retries = 3
    for attempt in range(retries):
        try:
            async with session.request(
                "POST" if json_data else "GET",
                url,
                params=params,
                headers=headers,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    continue
                else:
                    logger.warning(f"HTTP {resp.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url} (attempt {attempt+1})")
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            await asyncio.sleep(2 ** attempt)
    return None

# -------------------------- Data Collector (Async) --------------------------
class AsyncDataCollector:
    def __init__(self):
        self.twitter_bearer = TWITTER_BEARER_TOKEN
        self.reddit_client_id = REDDIT_CLIENT_ID
        self.reddit_client_secret = REDDIT_CLIENT_SECRET
        self.reddit_user_agent = REDDIT_USER_AGENT
        self.dexscreener_limit = 10
        self.twitter_rate_limiter = RateLimiter(50, 60)
        self.reddit_rate_limiter = RateLimiter(60, 60)
        self.dexscreener_rate_limiter = RateLimiter(30, 60)
        self._reddit_token = None
        self._reddit_token_expiry = None
        self._reddit_token_lock = asyncio.Lock()

    async def _get_reddit_oauth_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        async with self._reddit_token_lock:
            if self._reddit_token and self._reddit_token_expiry and datetime.now() < self._reddit_token_expiry:
                return self._reddit_token
            if not self.reddit_client_id or not self.reddit_client_secret:
                return None
            url = "https://www.reddit.com/api/v1/access_token"
            auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
            data = {"grant_type": "client_credentials"}
            try:
                async with session.post(url, auth=auth, data=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        self._reddit_token = result["access_token"]
                        self._reddit_token_expiry = datetime.now() + timedelta(seconds=result["expires_in"])
                        return self._reddit_token
                    else:
                        logger.warning(f"Reddit OAuth failed: {resp.status}")
            except Exception as e:
                logger.error(f"Reddit OAuth error: {e}")
            return None

    async def get_twitter_trending(self, session: aiohttp.ClientSession) -> List[str]:
        if not self.twitter_bearer:
            return []

        await self.twitter_rate_limiter.acquire()
        url = "https://api.twitter.com/2/trends/place"
        params = {"id": "1"}
        headers = {"Authorization": f"Bearer {self.twitter_bearer}"}
        data = await fetch_json(session, url, params=params, headers=headers, timeout=10)
        if data and "trends" in data and data["trends"]:
            return [trend["name"] for trend in data["trends"][:10]]

        logger.info("Trends endpoint unavailable, falling back to search-based trending")
        query = "-is:retweet lang:en"
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "public_metrics",
            # sort_order removed
        }
        data = await fetch_json(session, url, params=params, headers=headers, timeout=15)
        if data and "data" in data:
            topics = []
            for tweet in data["data"]:
                text = tweet.get("text", "")
                words = text.split()[:5]
                topic = " ".join(words).strip()
                if topic and len(topic) > 3:
                    topics.append(topic)
            return list(set(topics))[:10]

        logger.warning("Twitter trending fallback failed, using hardcoded keywords")
        return ["solana", "memecoin", "bitcoin", "ethereum", "nft", "ai", "degen", "pumpfun"]

    async def search_tweets(self, session: aiohttp.ClientSession, query: str, start_time: datetime, end_time: datetime) -> int:
        if not self.twitter_bearer:
            return 0
        await self.twitter_rate_limiter.acquire()
        url = "https://api.twitter.com/2/tweets/counts/recent"
        params = {
            "query": f"{query} lang:en -is:retweet",
            "start_time": start_time.isoformat(timespec="seconds") + "Z",
            "end_time": end_time.isoformat(timespec="seconds") + "Z",
            "granularity": "minute"
        }
        headers = {"Authorization": f"Bearer {self.twitter_bearer}"}
        data = await fetch_json(session, url, params=params, headers=headers, timeout=10)
        if data and "data" in data:
            total = sum(entry["tweet_count"] for entry in data["data"])
            # Save mention counts for history
            await save_mentions(query, "twitter", total)
            return total
        return 0

    async def get_reddit_posts(self, session: aiohttp.ClientSession, subreddits: List[str], limit: int = 10) -> List[Dict]:
        await self.reddit_rate_limiter.acquire()
        posts = []
        token = await self._get_reddit_oauth_token(session)
        for sub in subreddits:
            url = f"https://oauth.reddit.com/r/{sub}/hot" if token else f"https://www.reddit.com/r/{sub}/hot.json"
            params = {"limit": limit}
            headers = {"User-Agent": self.reddit_user_agent}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            data = await fetch_json(session, url, params=params, headers=headers, timeout=10)
            if data and "data" in data and "children" in data["data"]:
                for child in data["data"]["children"]:
                    post = child["data"]
                    posts.append({
                        "title": post["title"],
                        "score": post["score"],
                        "num_comments": post["num_comments"],
                        "created_utc": post["created_utc"],
                        "subreddit": sub
                    })
                    # Save mention counts (each post counts as one mention of its title)
                    await save_mentions(normalize_topic(post["title"]), "reddit", 1)
            await asyncio.sleep(0.5)
        return posts

    async def get_dexscreener_pairs(self, session: aiohttp.ClientSession, query: str, chain: str = "solana") -> List[Dict]:
        await self.dexscreener_rate_limiter.acquire()
        # Improved query: if topic looks like a ticker (short, all caps) or known crypto ticker
        known_tickers = {"BTC", "ETH", "SOL", "DOGE", "SHIB", "PEPE", "WIF", "BONK"}
        if (query.isupper() and len(query) <= 5) or query.upper() in known_tickers:
            search_query = f"${query}"
        else:
            search_query = query
        url = DEXSCREENER_API_URL
        params = {"q": search_query}
        data = await fetch_json(session, url, params=params, timeout=10)
        if not data or "pairs" not in data:
            return []
        pairs = data["pairs"]
        filtered = []
        cutoff = datetime.now() - timedelta(days=7)
        for pair in pairs:
            if pair.get("chainId") != chain:
                continue
            created_at = pair.get("pairCreatedAt")
            if created_at:
                created_time = datetime.fromtimestamp(created_at / 1000)
                if created_time < cutoff:
                    continue
            market_cap = pair.get("marketCap")
            if market_cap and market_cap > 5_000_000:
                filtered.append(pair)
        return filtered

# -------------------------- LLM Integration (Async) --------------------------
# Base class for all LLM providers
class BaseLLM:
    async def ask(self, session: aiohttp.ClientSession, prompt: str, system: str = None, timeout: int = 30) -> str:
        raise NotImplementedError

    async def generate_coin_concept(self, session: aiohttp.ClientSession, topic: str, scores: Dict) -> Dict:
        prompt = f"""
Topic: {topic}
Pump Score: {scores['pump_score']:.0f}/100
Virality: {scores['virality']:.0f}/40
Memeability: {scores['memeability']:.0f}/30
Novelty: {scores['novelty']:.0f}/20
Crypto Momentum: {scores['crypto_momentum']:.0f}/10

Create a meme coin concept:
1. Catchy name (3-8 letters, punny)
2. Ticker ($XXXX)
3. 2-sentence hype description (funny + community CTA + tokenomics tease)
4. Why it will 100x (3-5 bullet points)
5. Logo generation prompt for Grok Imagine / Midjourney (detailed)

Avoid DOGE/PEPE/SHIB/WIF clones. Focus on fresh 2026 themes.

Return as JSON:
{{
    "name": "...",
    "ticker": "...",
    "description": "...",
    "why_100x": ["...", "..."],
    "logo_prompt": "..."
}}
"""
        system = "You are a meme coin strategist. Output valid JSON."
        response = await self.ask(session, prompt, system, timeout=15)
        if response:
            data = extract_json_from_response(response)
            if data:
                return {
                    "name": data.get("name", f"{topic.capitalize()}Coin"),
                    "ticker": data.get("ticker", f"${topic.upper()[:5]}"),
                    "description": data.get("description", "To the moon!"),
                    "why_100x": data.get("why_100x", ["Viral potential", "Strong community", "Meme power"]),
                    "logo_prompt": data.get("logo_prompt", f"A cute {topic} meme coin logo")
                }
        # Fallback
        logger.warning(f"Using fallback concept for {topic}")
        return {
            "name": f"{topic.capitalize()}",
            "ticker": f"${topic.upper()[:5]}",
            "description": f"Get ready for {topic} to moon! Join the community now.",
            "why_100x": ["Viral potential", "Strong community", "Meme power"],
            "logo_prompt": f"Cartoon {topic} character with crypto elements"
        }

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.rate_limiter = RateLimiter(30, 60)

    async def ask(self, session: aiohttp.ClientSession, prompt: str, system: str = None, timeout: int = 30) -> str:
        await self.rate_limiter.acquire()
        # Gemini API uses POST to https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        if system:
            # Gemini doesn't have system prompt; we can prepend it to the user prompt.
            payload["contents"][0]["parts"][0]["text"] = f"{system}\n\n{prompt}"
        headers = {"Content-Type": "application/json"}
        data = await fetch_json(session, url, json_data=payload, headers=headers, timeout=timeout)
        if data and "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            logger.error(f"Gemini error: {data}")
            return ""

class OpenAIChatLLM(BaseLLM):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.rate_limiter = RateLimiter(30, 60)

    async def ask(self, session: aiohttp.ClientSession, prompt: str, system: str = None, timeout: int = 30) -> str:
        await self.rate_limiter.acquire()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7
        }
        url = f"{self.base_url}/chat/completions"
        data = await fetch_json(session, url, json_data=payload, headers=headers, timeout=timeout)
        if data and "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"LLM error: {data}")
            return ""

# -------------------------- Scoring Engine (Async) --------------------------
class AsyncScoringEngine:
    def __init__(self, collector: AsyncDataCollector, llm: BaseLLM):
        self.collector = collector
        self.llm = llm

    async def calculate_virality(self, session: aiohttp.ClientSession, topic: str) -> float:
        end = datetime.now()
        start_60 = end - timedelta(minutes=60)
        start_24h = end - timedelta(hours=24)
        count_60 = await self.collector.search_tweets(session, topic, start_60, end)
        count_24h = await self.collector.search_tweets(session, topic, start_24h, end)
        if count_24h == 0:
            return 0
        avg = count_24h / 24
        ratio = count_60 / avg if avg > 0 else 0
        virality = ratio * 10
        return min(virality, 40)

    async def calculate_memeability(self, session: aiohttp.ClientSession, topic: str) -> float:
        prompt = f"Rate the memeability of the topic '{topic}' on a scale of 0-10 (0=not memey, 10=extremely viral and memeable). Return only the number."
        response = await self.llm.ask(session, prompt, timeout=5)
        try:
            num = re.search(r'\d+', response)
            if num:
                score = int(num.group())
                score = max(0, min(score, 10))
                return score * 3
        except:
            pass
        return 0

    async def calculate_novelty(self, session: aiohttp.ClientSession, topic: str) -> float:
        similar = await self.collector.get_dexscreener_pairs(session, topic, chain="solana")
        penalty = min(len(similar) * 1, 20)
        return 20 - penalty

    async def calculate_crypto_momentum(self, session: aiohttp.ClientSession, topic: str) -> float:
        crypto_keywords = ["solana", "sol", "ethereum", "eth", "bitcoin", "btc", "memecoin", "airdrop", "degen"]
        if any(kw in topic.lower() for kw in crypto_keywords):
            return 10
        return 0

    async def calculate_pump_score(self, session: aiohttp.ClientSession, topic: str) -> Dict:
        virality_task = asyncio.create_task(self.calculate_virality(session, topic))
        memeability_task = asyncio.create_task(self.calculate_memeability(session, topic))
        novelty_task = asyncio.create_task(self.calculate_novelty(session, topic))
        crypto_momentum_task = asyncio.create_task(self.calculate_crypto_momentum(session, topic))

        virality, memeability, novelty, crypto_momentum = await asyncio.gather(
            virality_task, memeability_task, novelty_task, crypto_momentum_task
        )

        pump_score = virality + memeability + novelty + crypto_momentum
        return {
            "virality": virality,
            "memeability": memeability,
            "novelty": novelty,
            "crypto_momentum": crypto_momentum,
            "pump_score": pump_score
        }

# -------------------------- Telegram Bot (Async) --------------------------
class TelegramBot:
    def __init__(self, token: str, collector: AsyncDataCollector, engine: AsyncScoringEngine, llm: BaseLLM):
        self.token = token
        self.collector = collector
        self.engine = engine
        self.llm = llm
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        self._stop = False

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("analyze", self.analyze))
        self.application.add_handler(CommandHandler("set_threshold", self.set_threshold))
        self.application.add_handler(CommandHandler("force_alert", self.force_alert))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 Meme Coin Trend Analyzer Bot active!\n"
            "Commands:\n"
            "/analyze - Manual trend scan\n"
            "/set_threshold <score> - Change alert threshold\n"
            "/force_alert <topic> - Force alert for a topic"
        )

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Analyzing trends... (this may take a minute)")
        async with aiohttp.ClientSession() as session:
            trending = await self.collector.get_twitter_trending(session)
            if not trending:
                await update.message.reply_text("No trending topics found. Check API keys.")
                return
            results = []
            for topic in trending[:5]:
                scores = await self.engine.calculate_pump_score(session, topic)
                results.append((topic, scores["pump_score"]))
            msg = "Top trends with scores:\n"
            for topic, score in results:
                msg += f"\n{topic[:50]}: {score:.0f}/100"
            await update.message.reply_text(msg[:4096])

    async def set_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        try:
            new_thresh = int(context.args[0])
            await set_user_threshold(chat_id, new_thresh)
            await update.message.reply_text(f"Threshold set to {new_thresh} for this chat.")
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /set_threshold <score>")

    async def force_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /force_alert <topic>")
            return
        topic = " ".join(context.args)
        async with aiohttp.ClientSession() as session:
            scores = await self.engine.calculate_pump_score(session, topic)
            generated = await self.llm.generate_coin_concept(session, topic, scores)
            alert_text = self.format_alert(topic, scores, generated)
            await update.message.reply_text(alert_text, parse_mode="Markdown")

    def format_alert(self, topic: str, scores: Dict, generated: Dict) -> str:
        lines = [
            f"*🚀 NEW PUMP SIGNAL: {topic.upper()}*",
            f"*Pump Score:* {scores['pump_score']:.0f}/100",
            "",
            "*Score Breakdown:*",
            f"🔥 Virality: {scores['virality']:.0f}/40",
            f"😂 Memeability: {scores['memeability']:.0f}/30",
            f"✨ Novelty: {scores['novelty']:.0f}/20",
            f"📈 Crypto Momentum: {scores['crypto_momentum']:.0f}/10",
            "",
            "*Coin Concept:*",
            f"Name: {generated['name']}",
            f"Ticker: {generated['ticker']}",
            f"Description: {generated['description']}",
            "",
            "*Why it will 100x:*",
        ]
        for bullet in generated['why_100x']:
            lines.append(f"• {bullet}")
        lines.append("")
        lines.append(f"*Logo Prompt:* `{generated['logo_prompt']}`")
        lines.append("")
        lines.append("_Launch template ready!_")
        return "\n".join(lines)

    async def send_alert(self, chat_id: int, topic: str, scores: Dict, generated: Dict):
        text = self.format_alert(topic, scores, generated)
        await self.application.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")

    async def send_error(self, error_message: str):
        chat_id = ERROR_CHAT_ID
        if not chat_id:
            return
        if len(error_message) > 4000:
            file = BytesIO(error_message.encode())
            file.name = "error.txt"
            await self.application.bot.send_document(chat_id=chat_id, document=InputFile(file), caption="Error details")
        else:
            await self.application.bot.send_message(chat_id=chat_id, text=f"⚠️ *ERROR*: {error_message}", parse_mode="Markdown")

    async def daily_summary(self, context: ContextTypes.DEFAULT_TYPE):
        """Send daily summary of top topics from last 24 hours."""
        chat_id = DEFAULT_CHAT_ID
        if not chat_id:
            return
        rows = await get_topics_last_24h(limit=10)
        if not rows:
            await self.application.bot.send_message(chat_id=chat_id, text="📊 *Daily Summary*\nNo topics detected in the last 24 hours.", parse_mode="Markdown")
            return
        summary = "📊 *Daily Summary (Last 24h)*\n\n"
        for i, row in enumerate(rows, 1):
            topic, score, created_at, name, ticker = row
            summary += f"{i}. *{topic}* – Score: {score:.0f}\n"
            if name and ticker:
                summary += f"   🪙 {name} ({ticker})\n"
            summary += "\n"
        summary += "_Powered by Meme Coin Trend Analyzer_"
        await self.application.bot.send_message(chat_id=chat_id, text=summary[:4096], parse_mode="Markdown")

    async def polling_job(self, context: ContextTypes.DEFAULT_TYPE):
        if self._stop:
            return
        logger.info("Running polling job...")
        chat_id = DEFAULT_CHAT_ID
        if not chat_id:
            logger.warning("No TELEGRAM_CHAT_ID set; alerts will not be sent.")
            return
        try:
            async with aiohttp.ClientSession() as session:
                twitter_trending = await self.collector.get_twitter_trending(session)
                reddit_posts = await self.collector.get_reddit_posts(session, ["memecoins", "cryptocurrency", "wallstreetbets"], limit=5)
                reddit_topics = [post["title"] for post in reddit_posts]
                all_topics = set()
                for t in twitter_trending + reddit_topics:
                    normalized = normalize_topic(t)
                    if normalized:
                        all_topics.add(normalized)
                all_topics = list(all_topics)[:10]
                for topic in all_topics:
                    last_score = await get_recent_alert_score(topic)
                    scores = await self.engine.calculate_pump_score(session, topic)
                    pump_score = scores["pump_score"]
                    threshold = await get_user_threshold(int(chat_id))
                    if pump_score >= threshold:
                        if last_score is None or (pump_score - last_score) >= 15:
                            generated = await self.llm.generate_coin_concept(session, topic, scores)
                            topic_id = await save_topic(topic, scores, generated, alert=True)
                            await self.send_alert(int(chat_id), topic, scores, generated)
                            logger.info(f"Alert sent for {topic} (score {pump_score:.0f})")
                        else:
                            logger.info(f"Skipping {topic}: cooldown active (last score {last_score})")
                    else:
                        await save_topic(topic, scores, generated=None, alert=False)
        except Exception as e:
            error_msg = f"Polling job failed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            await self.send_error(error_msg)

    def schedule_polling(self):
        self.job_queue = self.application.job_queue
        if self.job_queue:
            self.job_queue.run_repeating(self.polling_job, interval=POLL_INTERVAL_MINUTES * 60, first=10)

    def schedule_daily_summary(self):
        tz = pytz.timezone(DAILY_SUMMARY_TIMEZONE)
        now = datetime.now(tz)
        target = now.replace(hour=DAILY_SUMMARY_HOUR, minute=DAILY_SUMMARY_MINUTE, second=0, microsecond=0)
        if now > target:
            target += timedelta(days=1)
        delta = (target - now).total_seconds()
        self.job_queue.run_once(self.daily_summary, delta)
        # Also schedule daily recurring
        self.job_queue.run_daily(self.daily_summary, time=target.time(), days=tuple(range(7)))

    async def start_bot(self):
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        self.schedule_polling()
        self.schedule_daily_summary()
        logger.info("Telegram bot started")

    async def stop_bot(self):
        self._stop = True
        if self.application.job_queue:
            for job in self.application.job_queue.jobs():
                job.schedule_removal()
        if self.application.updater:
            await self.application.updater.stop()
        await self.application.stop()
        logger.info("Telegram bot stopped")

# -------------------------- FastAPI Dashboard --------------------------
fastapi_app = FastAPI(title="Meme Coin Trend Analyzer")

@fastapi_app.get("/", response_class=HTMLResponse)
async def dashboard():
    rows = await get_latest_scores()
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Meme Coin Trend Analyzer</title></head>
    <body>
        <h1>Latest Trends</h1>
        <table border="1">
            <thead><tr><th>Topic</th><th>Pump Score</th><th>Time</th><th>Generated Name</th><th>Ticker</th></tr></thead>
            <tbody>
    """
    for row in rows:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>"
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@fastapi_app.get("/api/latest")
async def api_latest():
    rows = await get_latest_scores()
    data = [{"topic": r[0], "score": r[1], "time": r[2], "name": r[3], "ticker": r[4]} for r in rows]
    return JSONResponse(content=data)

@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}

async def run_fastapi():
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

# -------------------------- Main Entry Point --------------------------
async def main():
    await init_db()
    collector = AsyncDataCollector()

    # Select LLM provider
    llm = None
    if GEMINI_API_KEY:
        llm = GeminiLLM(GEMINI_API_KEY, GEMINI_MODEL)
        logger.info(f"Using Gemini (model: {GEMINI_MODEL}) as primary LLM")
    elif GROK_API_KEY:
        llm = OpenAIChatLLM(GROK_API_KEY, GROK_API_BASE, GROK_MODEL)
        logger.info(f"Using Grok (model: {GROK_MODEL}) as primary LLM")
    elif OPENAI_API_KEY:
        llm = OpenAIChatLLM(OPENAI_API_KEY, model=OPENAI_MODEL)
        logger.info(f"Using OpenAI (model: {OPENAI_MODEL}) as fallback LLM")
    else:
        logger.warning("No LLM API key set; using dummy generator")
        class DummyLLM(BaseLLM):
            async def ask(self, session, prompt, system=None, timeout=30):
                return "5"
        llm = DummyLLM()

    engine = AsyncScoringEngine(collector, llm)
    bot = TelegramBot(TELEGRAM_BOT_TOKEN, collector, engine, llm)

    # Start bot and FastAPI concurrently
    bot_task = asyncio.create_task(bot.start_bot())
    api_task = asyncio.create_task(run_fastapi())

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Signal received, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await stop_event.wait()
    logger.info("Initiating graceful shutdown...")
    bot_task.cancel()
    api_task.cancel()
    await asyncio.gather(bot_task, api_task, return_exceptions=True)
    await bot.stop_bot()
    logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
