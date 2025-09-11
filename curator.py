import os
import feedparser
import requests
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from jinja2 import Template
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from bs4 import BeautifulSoup
import logging

# === CONFIG ===
SOURCES = [
    {"name": "Reuters", "url": "http://feeds.reuters.com/reuters/worldNews", "weight": 1.0},
    {"name": "AP News", "url": "https://rsshub.app/apnews/topics/world", "weight": 1.0},
    {"name": "BBC", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "weight": 0.9},
    {"name": "The Guardian", "url": "https://www.theguardian.com/world/rss", "weight": 0.8},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "weight": 0.85},
]

CATEGORIES = [
    "World",
    "Politics",
    "Technology",
    "Science",
    "Business",
    "Health",
    "Environment",
    "Sports",
    "Culture"
]

TOP_N = 5
DEDUP_THRESHOLD = 0.85

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === AI MODELS (load once) ===
logger.info("Loading AI models...")
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("AI models loaded.")
except Exception as e:
    logger.error(f"Failed to load AI models: {e}")
    exit(1)

# === FETCH & PARSE ARTICLES ===
def fetch_articles():
    articles = []
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=48)

    for source in SOURCES:
        try:
            logger.info(f"Fetching from {source['name']}...")
            feed = feedparser.parse(source['url'])
            for entry in feed.entries:
                # Parse and normalize publish date
                try:
                    if hasattr(entry, 'published'):
                        pub_date = date_parser.parse(entry.published)
                        if pub_date.tzinfo is not None:
                            pub_date = pub_date.replace(tzinfo=None)  # Make naive
                    else:
                        pub_date = now
                except Exception as e:
                    logger.error(f"Failed to parse date for {entry.title}: {e}")
                    pub_date = now

                # Skip if older than 48h
                if pub_date < cutoff:
                    continue

                # Extract summary
                summary = getattr(entry, 'summary', '')
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                if not summary and hasattr(entry, 'content'):
                    try:
                        summary = BeautifulSoup(entry.content[0].value, "html.parser").get_text()[:300] + "..."
                    except Exception as e:
                        logger.warning(f"Could not extract content: {e}")
                        summary = ""

                articles.append({
                    "title": entry.title,
                    "summary": summary,
                    "url": entry.link,
                    "source": source["name"],
                    "source_weight": source["weight"],
                    "published_at": pub_date,
                    "text_for_ai": f"{entry.title}. {summary}"
                })
        except Exception as e:
            logger.error(f"Error fetching {source['name']}: {e}")

    logger.info(f"Fetched {len(articles)} articles.")
    return articles

# === CLASSIFY ARTICLES ===
def classify_articles(articles):
    for article in articles:
        try:
            result = classifier(article["text_for_ai"], CATEGORIES, multi_label=False)
            article["category"] = result["labels"][0]
            article["category_confidence"] = result["scores"][0]
        except Exception as e:
            logger.error(f"Classification failed for '{article['title']}': {e}")
            article["category"] = "World"  # fallback
            article["category_confidence"] = 0.5
    return articles

# === CALCULATE IMPORTANCE SCORE ===
def calculate_score(article, now):
    score = article["source_weight"]

    # Recency decay
    age_hours = (now - article["published_at"]).total_seconds() / 3600
    recency_multiplier = max(0.1, 1 - (age_hours / 48))
    score *= recency_multiplier

    # Bonus for entity-rich content (rough proxy)
    entity_bonus = min(len(article["title"].split()) * 0.02, 0.3)
    score += entity_bonus

    # Confidence bonus
    score *= article["category_confidence"]

    article["score"] = score
    return score

# === DEDUPLICATE ARTICLES ===
def deduplicate_articles(articles):
    if len(articles) == 0:
        return []

    texts = [a["title"] + " " + a["summary"][:100] for a in articles]
    try:
        embeddings = embedder.encode(texts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return articles  # fallback: no deduplication

    to_remove = set()
    n = len(articles)

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if cosine_scores[i][j] > DEDUP_THRESHOLD:
                if articles[i]["score"] >= articles[j]["score"]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

    deduped = [a for i, a in enumerate(articles) if i not in to_remove]
    logger.info(f"Deduplicated: {len(articles)} → {len(deduped)} articles.")
    return deduped

# === RANK & SELECT TOP N PER CATEGORY ===
def select_top_per_category(articles):
    categorized = {cat: [] for cat in CATEGORIES}
    now = datetime.utcnow()

    for article in articles:
        categorized[article["category"]].append(article)

    top_articles = {}
    for cat, articles_in_cat in categorized.items():
        # Sort by score descending
        sorted_articles = sorted(articles_in_cat, key=lambda x: x.get("score", 0), reverse=True)
        top_articles[cat] = sorted_articles[:TOP_N]

    return top_articles

# === GENERATE HTML ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quiet.News — Top News. No Noise.</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }
        h1 { color: #222; border-bottom: 3px solid #eee; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 30px; }
        li { margin-bottom: 20px; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .summary { color: #555; margin: 5px 0; }
        .meta { font-size: 0.9em; color: #888; }
        footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #888; }
    </style>
</head>
<body>
    <h1>Quiet.News</h1>
    <p><em>Top news. No ads. No noise. Updated automatically.</em></p>

    {% for category, articles in categorized.items() %}
        {% if articles|length > 0 %}
        <h2>{{ category }}</h2>
        <ol>
        {% for article in articles %}
            <li>
                <strong>{{ article.title }}</strong><br>
                <div class="summary">{{ article.summary }}</div>
                <div class="meta">→ {{ article.source }} | Score: {{ "%.2f"|format(article.score) }}</div>
                <a href="{{ article.url }}" target="_blank">Read full</a>
            </li>
        {% endfor %}
        </ol>
        {% endif %}
    {% endfor %}

    <footer>
        <p>Updated: {{ now.strftime('%Y-%m-%d %H:%M UTC') }} | Sources: Reuters, AP, BBC, Guardian, Al Jazeera</p>
    </footer>
</body>
</html>
"""

def generate_html(categorized):
    template = Template(HTML_TEMPLATE)
    html = template.render(
        categorized=categorized,
        now=datetime.utcnow()
    )
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML generated: index.html")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    logger.info("Starting automated news curator...")
    articles = fetch_articles()
    if not articles:
        logger.error("No articles fetched. Exiting.")
        exit(1)

    articles = classify_articles(articles)

    # ✅ Calculate scores BEFORE deduplication
    now = datetime.utcnow()
    for article in articles:
        calculate_score(article, now)

    articles = deduplicate_articles(articles)
    categorized = select_top_per_category(articles)
    generate_html(categorized)
    logger.info("✅ Done. Website updated.")
