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
    {"name": "The Hindu", "url": "https://www.thehindu.com/feeder/default.rss", "weight": 1.0},
    {"name": "Indian Express", "url": "https://indianexpress.com/section/india/feed/", "weight": 0.95},
    {"name": "NDTV", "url": "https://feeds.feedburner.com/ndtvnews-india-news", "weight": 0.9},
    {"name": "Times of India", "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "weight": 0.85},
    {"name": "BBC News India", "url": "http://feeds.bbci.co.uk/news/world/asia/india/rss.xml", "weight": 0.85},
    {"name": "Reuters India", "url": "https://www.reuters.com/world/india/rss", "weight": 0.9},
    {"name": "Al Jazeera - Asia", "url": "https://www.aljazeera.com/xml/rss/all.xml", "weight": 0.8},
    {"name": "AP News - Asia", "url": "https://rsshub.app/apnews/topics/asia", "weight": 0.8},
]

CATEGORIES = [
    "India",
    "World",
    "Politics",
    "Technology",
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

# === HELPER: Detect India-related content ===
def is_india_related(text):
    india_keywords = [
        'india', 'indian', 'delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata',
        'modi', 'bjp', 'congress', 'aadhaar', 'upi', 'india gdp', 'india economy',
        'supreme court india', 'india election', 'lok sabha', 'rajya sabha',
        'india china', 'india pakistan', 'jammu', 'kashmir', 'gujarat', 'tamil nadu',
        'bihar', 'uttar pradesh', 'maharashtra', 'karnataka', 'telangana', 'andhra',
        'punjab', 'haryana', 'rajasthan', 'assam', 'bengal', 'odisha', 'kerela',
        'indian rupee', 'rbi', 'sebi', 'nifty', 'sensex'
    ]
    text_lower = text.lower()
    for kw in india_keywords:
        if kw in text_lower:
            return True
    return False

# === HELPER: Editorial Dimensions ===
def is_trending(text):
    trending_keywords = [
        'viral', 'trending', 'breaking', 'explainer', 'chart', 'graph', 'spike',
        'record high', 'suddenly', 'overnight', 'everyone is talking about',
        'blows up', 'skyrockets', 'plummets', 'goes viral', 'tops charts',
        'most watched', 'most searched', 'google trends', 'twitter trends'
    ]
    text_lower = text.lower()
    for kw in trending_keywords:
        if kw in text_lower:
            return True
    return False

def is_debatable(text):
    debatable_keywords = [
        'controversy', 'debate', 'divided', 'clash', 'protests', 'backlash',
        'criticism', 'defends', 'slammed', 'outrage', 'calls for', 'demands',
        'should', 'must', 'why', 'how could', 'scandal', 'allegations',
        'court battle', 'legal fight', 'ethics', 'morality', 'cancel culture',
        'free speech', 'censorship', 'bias', 'fake news'
    ]
    text_lower = text.lower()
    for kw in debatable_keywords:
        if kw in text_lower:
            return True
    return False

def is_must_know(text):
    must_know_keywords = [
        'new law', 'policy change', 'supreme court rules', 'election results',
        'major study', 'who should know', 'everyone needs to know', 'urgent',
        'critical', 'essential', 'what you need to know', 'implications',
        'long-term', 'affects everyone', 'national security', 'public health',
        'economy update', 'market crash', 'inflation', 'unemployment'
    ]
    text_lower = text.lower()
    for kw in must_know_keywords:
        if kw in text_lower:
            return True
    return False

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
                try:
                    if hasattr(entry, 'published'):
                        pub_date = date_parser.parse(entry.published)
                        if pub_date.tzinfo is not None:
                            pub_date = pub_date.replace(tzinfo=None)
                    else:
                        pub_date = now
                except Exception as e:
                    logger.error(f"Failed to parse date for {entry.title}: {e}")
                    pub_date = now

                if pub_date < cutoff:
                    continue

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
            if is_india_related(article["text_for_ai"]):
                article["category"] = "India"
                article["category_confidence"] = 1.0
            else:
                result = classifier(article["text_for_ai"], CATEGORIES, multi_label=False)
                article["category"] = result["labels"][0]
                article["category_confidence"] = result["scores"][0]

            # Tag with editorial dimensions
            article["is_trending"] = is_trending(article["text_for_ai"])
            article["is_debatable"] = is_debatable(article["text_for_ai"])
            article["is_must_know"] = is_must_know(article["text_for_ai"])

        except Exception as e:
            logger.error(f"Classification failed for '{article['title']}': {e}")
            article["category"] = "World"
            article["category_confidence"] = 0.5
            article["is_trending"] = False
            article["is_debatable"] = False
            article["is_must_know"] = False
    return articles

# === CALCULATE IMPORTANCE SCORE ===
def calculate_score(article, now):
    score = article["source_weight"]

    # Recency decay
    age_hours = (now - article["published_at"]).total_seconds() / 3600
    recency_multiplier = max(0.1, 1 - (age_hours / 48))
    score *= recency_multiplier

    # Entity bonus
    entity_bonus = min(len(article["title"].split()) * 0.02, 0.3)
    score += entity_bonus

    # Confidence bonus
    score *= article["category_confidence"]

    # 🇮🇳 India relevance boost
    if is_india_related(article["text_for_ai"]):
        score *= 1.5

    # 📈 Editorial dimension boosts
    if article.get("is_trending", False):
        score *= 1.3
    if article.get("is_debatable", False):
        score *= 1.2
    if article.get("is_must_know", False):
        score *= 1.5

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
        return articles

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
    logger.info(f"Deduplicated: {len(articles)} -> {len(deduped)} articles.")
    return deduped

# === RANK & SELECT TOP N PER CATEGORY ===
def select_top_per_category(articles):
    categorized = {cat: [] for cat in CATEGORIES}
    now = datetime.utcnow()

    for article in articles:
        calculate_score(article, now)
        categorized[article["category"]].append(article)

    top_articles = {}
    for cat, articles_in_cat in categorized.items():
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
    <title>Quiet.News - Timeless News for India</title>
    <style>
        /* === BASE === */
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            background: url('https://www.transparenttextures.com/patterns/cream-paper.png'), #f8f4e9;
            background-attachment: fixed;
            color: #3a3a3a;
            max-width: 900px;
            margin: 0 auto;
            padding: 30px 20px;
            line-height: 1.6;
        }

        /* === HEADER === */
        header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #c9b89b;
        }

        .logo {
            max-width: 350px;
            margin: 0 auto 15px;
            display: block;
            filter: contrast(1.1) brightness(0.9);
            transition: transform 0.2s ease;
        }

        .logo:hover {
            transform: scale(1.02);
        }

        h1 {
            font-family: 'Old Standard TT', 'Times New Roman', serif;
            font-weight: normal;
            letter-spacing: 2px;
            font-size: 2.4rem;
            color: #2c1e1e;
            margin: 10px 0 5px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        .tagline {
            font-style: italic;
            color: #6b5c45;
            font-size: 1.1rem;
            margin: 0;
        }

        /* === CATEGORY HEADINGS === */
        h2 {
            font-family: 'Old Standard TT', Georgia, serif;
            color: #2c1e1e;
            border-left: 5px solid #c9b89b;
            padding-left: 15px;
            margin: 40px 0 20px;
            font-size: 1.6rem;
            letter-spacing: 1px;
        }

        /* === ARTICLE CARD === */
        .article-card {
            background: #fffaf2;
            border: 1px solid #e0d5c1;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: box-shadow 0.2s ease;
        }

        .article-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        .article-title {
            font-family: 'Old Standard TT', Georgia, serif;
            font-size: 1.3rem;
            font-weight: normal;
            margin: 0 0 10px;
            line-height: 1.3;
            color: #222;
        }

        .article-summary {
            font-size: 1rem;
            color: #444;
            margin: 10px 0;
            text-align: justify;
        }

        .meta {
            font-size: 0.9rem;
            color: #7a6c5d;
            margin: 15px 0 10px;
            font-style: italic;
        }

        /* === TAGS === */
        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 5px;
            color: white;
        }

        .must-know { background: #8b0000; border: 1px solid #5a0000; }
        .trending { background: #004080; border: 1px solid #00264d; }
        .debatable { background: #b35900; border: 1px solid #7a3d00; }

        /* === LINK === */
        a {
            color: #004080;
            text-decoration: none;
            font-weight: bold;
            border-bottom: 1px dotted #004080;
            padding-bottom: 2px;
        }

        a:hover {
            color: #00264d;
            border-bottom-style: solid;
        }

        /* === FOOTER === */
        footer {
            margin-top: 60px;
            padding-top: 25px;
            border-top: 2px solid #c9b89b;
            color: #6b5c45;
            font-size: 0.95rem;
            text-align: center;
        }

        /* === TYPOGRAPHY ENHANCEMENTS === */
        @import url('https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=swap');

        /* === PRINT-LIKE TEXTURE === */
        .article-card {
            background-image: linear-gradient(rgba(255,255,255,0.8), rgba(250,245,235,0.8));
        }
    </style>
</head>
<body>
    <header>
        <img src="/logo.png" alt="Quiet.News Logo" class="logo">
        <h1>Quiet.News</h1>
        <p class="tagline">Timeless news for India. No ads. No noise.</p>
    </header>

    {% for category, articles in categorized.items() %}
        {% if articles|length > 0 %}
        <h2>{{ category }}</h2>
        {% for article in articles %}
        <div class="article-card">
            <div class="article-title">{{ article.title }}</div>
            <div class="article-summary">{{ article.summary }}</div>
            <div class="meta">
                → {{ article.source }}
                {% if article.is_must_know %}<span class="tag must-know">Must-Know</span>{% endif %}
                {% if article.is_trending %}<span class="tag trending">Trending</span>{% endif %}
                {% if article.is_debatable %}<span class="tag debatable">Debatable</span>{% endif %}
            </div>
            <a href="{{ article.url }}" target="_blank">Read full →</a>
        </div>
        {% endfor %}
        {% endif %}
    {% endfor %}

    <footer>
        <p>Curated and updated automatically • {{ now.strftime('%A, %B %d, %Y') }}</p>
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

    now = datetime.utcnow()
    for article in articles:
        calculate_score(article, now)

    articles = deduplicate_articles(articles)
    categorized = select_top_per_category(articles)
    generate_html(categorized)
    logger.info("✅ Done. Website updated.")
