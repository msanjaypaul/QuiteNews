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
    # 🎓 SCHOLARSHIPS
    {"name": "Scholarships in India", "url": "https://www.scholarshipsinindia.com/feed/", "weight": 1.0},
    {"name": "Buddy4Study", "url": "https://www.buddy4study.com/blog/feed/", "weight": 0.95},
    {"name": "Vidya Lakshmi", "url": "https://www.vidyalakshmi.co.in/NewsFeed", "weight": 0.9},
    {"name": "National Scholarship Portal", "url": "https://scholarships.gov.in/NewsRSS", "weight": 1.0},
    {"name": "AICTE Scholarships", "url": "https://www.aicte-india.org/scholarships/rss", "weight": 0.9},
    {"name": "UP Scholarship", "url": "https://scholarship.up.nic.in/RSSFeed.aspx", "weight": 0.85},
    {"name": "DAAD India", "url": "https://www.daad.in/en/rss/", "weight": 0.85},

    # 💼 INTERNSHIPS & JOBS
    {"name": "Internshala", "url": "https://internshala.com/blog/feed/", "weight": 1.0},
    {"name": "LetsIntern", "url": "https://www.letsintern.com/blog/feed/", "weight": 0.95},
    {"name": "Twenty19", "url": "https://twenty19.com/blog/feed", "weight": 0.9},
    {"name": "Hello Intern", "url": "https://www.hellointern.com/blog/feed/", "weight": 0.9},
    {"name": "Naukri Campus", "url": "https://www.naukri.com/campus-recruitment-blog/feed", "weight": 0.95},
    {"name": "Freshersworld", "url": "https://www.freshersworld.com/rss/jobs", "weight": 0.9},
    {"name": "Indeed Campus", "url": "https://www.indeed.com/career-advice/feed", "weight": 0.85},

    # 🏛 GOVERNMENT JOBS
    {"name": "Sarkari Naukri", "url": "https://www.sarkarinaukri.com/feed/", "weight": 1.0},
    {"name": "Sarkari Result", "url": "https://www.sarkariresult.com/rss", "weight": 0.95},
    {"name": "FreeJobAlert", "url": "https://www.freejobalert.com/feed/", "weight": 0.95},
    {"name": "SSC Adda", "url": "https://www.sscadda.com/feeds/posts/default", "weight": 0.9},
    {"name": "Bankers Adda", "url": "https://www.bankersadda.com/feeds/posts/default", "weight": 0.9},
    {"name": "UPSC Adda", "url": "https://www.upscadda.com/feeds/posts/default", "weight": 0.9},

    # 🎓 EXAMS & RESULTS
    {"name": "CBSE Latest", "url": "https://cbse.gov.in/cbsenew/rss.xml", "weight": 1.0},
    {"name": "NTA (JEE/NEET)", "url": "https://nta.ac.in/rss", "weight": 1.0},
    {"name": "UGC NET", "url": "https://ugcnet.nta.nic.in/rss", "weight": 0.95},
    {"name": "GATE", "url": "https://gate.iitk.ac.in/rss.xml", "weight": 0.9},

    # 🏫 COLLEGES & CAMPUS
    {"name": "DU Latest", "url": "https://www.du.ac.in/rss.xml", "weight": 0.9},
    {"name": "IIT Delhi", "url": "https://home.iitd.ac.in/rss.php", "weight": 0.85},
    {"name": "The Hindu - Education", "url": "https://www.thehindu.com/education/feeder/default.rss", "weight": 0.9},
    {"name": "Indian Express - Education", "url": "https://indianexpress.com/section/education/feed/", "weight": 0.95},
]

CATEGORIES = [
    "Scholarships",
    "Internships & Jobs",
    "Government Jobs",
    "Exams & Results",
    "Campus Life",
    "Mental Health",
    "Student Startups",
    "Education Policy",
    "Study Abroad",
    "Career Advice",
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

# === HELPER: Detect Student-Related Content ===
def is_student_related(text):
    text_lower = text.lower()

    strong_keywords = [
        'student', 'students', 'college', 'university', 'campus', 'exam', 'exams', 'result', 'results',
        'jee', 'neet', 'cat', 'gate', 'upsc', 'internship', 'placement', 'scholarship', 'admit card',
        'iit', 'nit', 'du', 'bhu', 'cbse', 'icse', 'board exam', 'cuet', 'clat', 'nda', 'aiims',
        'hostel', 'attendance', 'fee', 'protest', 'mental health', 'anxiety', 'stress', 'career',
        'resume', 'job', 'startup', 'founder', 'edtech', 'online class', 'syllabus', 'grade',
        'government job', 'sarkari naukri', 'apply online', 'notification', 'eligibility', 'salary'
    ]

    has_strong = any(kw in text_lower for kw in strong_keywords)

    reject_keywords = [
        'ireland', 'australia', 'uk', 'usa', 'canada', 'europe', 'global', 'international student',
        'south east technological university', 'tertiary education commission'
    ]

    has_reject = any(kw in text_lower for kw in reject_keywords)

    if has_reject and not ('india' in text_lower or 'indian' in text_lower):
        return False

    return has_strong

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
        'new law', 'policy change', 'supreme court rules', 'exam dates', 'admit card', 'result date',
        'major study', 'who should know', 'everyone needs to know', 'urgent',
        'critical', 'essential', 'what you need to know', 'implications',
        'long-term', 'affects everyone', 'national security', 'public health',
        'economy update', 'market crash', 'inflation', 'unemployment',
        'last date', 'apply now', 'notification released', 'exam postponed'
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
    cutoff = now - timedelta(days=7)  # ✅ Fetch from last 7 days (not 48h)

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
                            pub_date = pub_date.replace(tzinfo=None)
                    else:
                        pub_date = now
                except Exception as e:
                    logger.error(f"Failed to parse date for {entry.title}: {e}")
                    pub_date = now

                # Skip if older than 7 days
                if pub_date < cutoff:
                    continue

                # Extract title
                title = getattr(entry, 'title', '').strip()
                if not title:
                    continue

                # Extract summary
                summary = getattr(entry, 'summary', '')
                if not summary and hasattr(entry, 'content'):
                    try:
                        summary = BeautifulSoup(entry.content[0].value, "html.parser").get_text()
                    except:
                        pass

                # Clean and truncate summary
                summary = summary.replace('\n', ' ').strip()
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                if not summary:
                    summary = "Details inside — click to read full announcement."

                # Skip if summary is too short
                if len(summary) < 20:
                    continue

                # Extract image URL
                image_url = ""
                if hasattr(entry, 'media_content'):
                    for media in entry.media_content:
                        if media.type == 'image/jpeg' or media.type == 'image/png':
                            image_url = media.url
                            break
                elif hasattr(entry, 'enclosures'):
                    for enclosure in entry.enclosures:
                        if 'image' in enclosure.type:
                            image_url = enclosure.href
                            break

                articles.append({
                    "title": title,
                    "summary": summary,
                    "url": entry.link,
                    "source": source["name"],
                    "source_weight": source["weight"],
                    "published_at": pub_date,
                    "text_for_ai": f"{title}. {summary}",
                    "image_url": image_url
                })
        except Exception as e:
            logger.error(f"Error fetching {source['name']}: {e}")

    logger.info(f"Fetched {len(articles)} articles.")
    return articles

# === CLASSIFY ARTICLES ===
def classify_articles(articles):
    filtered_articles = []
    for article in articles:
        try:
            if is_student_related(article["text_for_ai"]):
                result = classifier(article["text_for_ai"], CATEGORIES, multi_label=False)
                article["category"] = result["labels"][0]
                article["category_confidence"] = result["scores"][0]

                # Force classification for key terms
                text_lower = article["text_for_ai"].lower()
                if any(kw in text_lower for kw in ['scholarship', 'vidyalakshmi', 'national scholarship']):
                    article["category"] = "Scholarships"
                elif any(kw in text_lower for kw in ['internship', 'intern', 'twenty19', 'hello intern']):
                    article["category"] = "Internships & Jobs"
                elif any(kw in text_lower for kw in ['sarkari naukri', 'government job', 'ssc', 'bankers adda', 'upsc adda']):
                    article["category"] = "Government Jobs"
                elif any(kw in text_lower for kw in ['exam', 'result', 'admit card', 'nta', 'cbse', 'gate', 'jee', 'neet']):
                    article["category"] = "Exams & Results"

                # Tag with editorial dimensions
                article["is_trending"] = is_trending(article["text_for_ai"])
                article["is_debatable"] = is_debatable(article["text_for_ai"])
                article["is_must_know"] = is_must_know(article["text_for_ai"])

                filtered_articles.append(article)
            else:
                continue
        except Exception as e:
            logger.error(f"Classification failed for '{article['title']}': {e}")
            article["category"] = "Campus Life"
            article["category_confidence"] = 0.5
            article["is_trending"] = False
            article["is_debatable"] = False
            article["is_must_know"] = False
            filtered_articles.append(article)
    return filtered_articles

# === CALCULATE IMPORTANCE SCORE ===
def calculate_score(article, now):
    score = article["source_weight"]

    # Recency decay (over 7 days)
    age_hours = (now - article["published_at"]).total_seconds() / 3600
    recency_multiplier = max(0.1, 1 - (age_hours / (7*24)))  # ✅ 7-day window
    score *= recency_multiplier

    # Entity bonus
    entity_bonus = min(len(article["title"].split()) * 0.02, 0.3)
    score += entity_bonus

    # Confidence bonus
    score *= article["category_confidence"]

    # 🎓 Student relevance boost
    if is_student_related(article["text_for_ai"]):
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
    logger.info(f"Deduplicated: {len(articles)} → {len(deduped)} articles.")
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  <!-- ✅ Mobile Responsive -->
    <title>Student.News — Scholarships, Internships, Govt Jobs</title>
    <style>
        /* === BASE === */
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            background: #f8f4e9;
            color: #2c1e1e;
            max-width: 1200px;
            margin: 0 auto;
            padding: 15px;  /* ✅ Reduced padding for mobile */
            line-height: 1.6;
        }

        /* === HEADER === */
        .masthead {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #2c1e1e;
            margin-bottom: 20px;
        }

        .title {
            font-family: 'Old Standard TT', serif;
            font-size: 2.5rem;  /* ✅ Smaller on mobile */
            font-weight: bold;
            letter-spacing: 1px;
            margin: 10px 0;
            color: #2c1e1e;
        }

        .tagline {
            font-size: 1rem;
            margin: 5px 0;
            color: #6b5c45;
            font-style: italic;
        }

        .date {
            font-size: 1rem;
            margin: 10px 0;
            color: #2c1e1e;
        }

        /* === QUOTE === */
        .quote-of-the-day {
            font-style: italic;
            font-size: 1.1rem;
            color: #6b5c45;
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border-top: 1px solid #ccc;
        }

        /* === CATEGORY === */
        .category-section {
            margin: 30px 0;
        }

        .category-title {
            font-size: 1.6rem;
            font-weight: bold;
            margin: 0 0 15px;
            color: #2c1e1e;
            border-bottom: 2px solid #c9b89b;
            padding-bottom: 5px;
        }

        /* === ARTICLE CARD === */
        .article-card {
            background: #fffaf2;
            border: 1px solid #e0d5c1;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .article-title {
            font-size: 1.3rem;
            font-weight: bold;
            margin: 0 0 10px;
            color: #222;
            line-height: 1.3;
        }

        .article-summary {
            font-size: 1rem;
            color: #444;
            margin: 10px 0;
            text-align: justify;
        }

        /* ✅ FIXED IMAGE DIMENSIONS + MOBILE RESPONSIVE */
        .article-card img {
            width: 100%;
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border-radius: 8px;
            object-fit: cover;
        }

        .meta {
            font-size: 0.9rem;
            color: #7a6c5d;
            margin: 15px 0 10px;
            font-style: italic;
        }

        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 6px;
            color: white;
        }

        .must-know { background: #8b0000; }
        .trending { background: #004080; }
        .debatable { background: #b35900; }

        a {
            color: #004080;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            border: 2px solid #004080;
            border-radius: 4px;
            transition: all 0.2s;
            font-size: 0.95rem;
        }

        a:hover {
            background: #004080;
            color: white;
        }

        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #c9b89b;
            color: #6b5c45;
            font-size: 0.9rem;
            text-align: center;
        }

        /* ✅ MOBILE OPTIMIZATION */
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            .title {
                font-size: 2rem;
            }
            .article-title {
                font-size: 1.2rem;
            }
            .category-title {
                font-size: 1.4rem;
            }
            .article-card {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="masthead">
        <h1 class="title">Student.News</h1>
        <p class="tagline">Scholarships, Internships, Govt Jobs. No fluff. No noise.</p>
        <p class="date">{{ now.strftime('%A, %B %d, %Y') }}</p>
    </div>

    <div class="quote-of-the-day">
        "Your future is created by what you do today, not tomorrow." — Robert Kiyosaki
    </div>

    {% for category, articles in categorized.items() %}
        {% if articles|length > 0 %}
        <div class="category-section">
            <h2 class="category-title">{{ category }}</h2>
            {% for article in articles %}
            <div class="article-card">
                {% if article.image_url %}
                    <img src="{{ article.image_url }}" alt="{{ article.title }}">
                {% endif %}
                <h3 class="article-title">{{ article.title }}</h3>
                <p class="article-summary">{{ article.summary }}</p>
                <div class="meta">
                    <em>Source: {{ article.source }}</em>
                    {% if article.is_must_know %}<span class="tag must-know">Must-Know</span>{% endif %}
                    {% if article.is_trending %}<span class="tag trending">Trending</span>{% endif %}
                    {% if article.is_debatable %}<span class="tag debatable">Debatable</span>{% endif %}
                </div>
                <a href="{{ article.url }}" target="_blank">📚 Read Full Article</a>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    {% endfor %}

    <footer>
        <p>Curated for Indian students • Updated: {{ now.strftime('%A, %B %d, %Y') }}</p>
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
    logger.info("Starting automated student news curator...")
    articles = fetch_articles()
    if not articles:
        logger.error("No articles fetched. Exiting.")
        exit(1)

    articles = classify_articles(articles)
    articles = deduplicate_articles(articles)

    now = datetime.utcnow()
    for article in articles:
        calculate_score(article, now)

    categorized = select_top_per_category(articles)
    generate_html(categorized)
    logger.info("✅ Done. Student.News updated.")
