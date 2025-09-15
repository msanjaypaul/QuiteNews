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
SCHOLARSHIP_SOURCES = [
    # GOVERNMENT & NATIONAL
    {"name": "National Scholarship Portal", "url": "https://scholarships.gov.in/NewsRSS", "weight": 1.0},
    {"name": "Vidya Lakshmi Portal", "url": "https://www.vidyalakshmi.co.in/NewsFeed", "weight": 0.95},
    {"name": "AICTE Scholarships", "url": "https://www.aicte-india.org/scholarships/rss", "weight": 0.95},
    {"name": "UGC Scholarships", "url": "https://nsp.ugc.ac.in/RSSFeed.aspx", "weight": 0.95},
    {"name": "Ministry of Minority Affairs", "url": "https://scholarships.gov.in/minority/rss", "weight": 0.9},

    # STATE SCHOLARSHIPS
    {"name": "UP Scholarship", "url": "https://scholarship.up.nic.in/RSSFeed.aspx", "weight": 0.9},
    {"name": "Bihar Scholarship", "url": "https://www.biharscholarship.in/RSSFeed.aspx", "weight": 0.9},
    {"name": "MP Scholarship", "url": "https://scholarshipportal.mp.nic.in/RSSFeed.aspx", "weight": 0.9},
    {"name": "Karnataka Scholarship", "url": "https://karepass.cgg.gov.in/rssFeed.jsp", "weight": 0.9},
    {"name": "Maharashtra Scholarship", "url": "https://mahadbt.maharashtra.gov.in/RSSFeed.aspx", "weight": 0.9},
    {"name": "Tamil Nadu Scholarship", "url": "https://www.tn.gov.in/scholarship/rss", "weight": 0.9},

    # CORPORATE & TRUSTS
    {"name": "Tata Trust Scholarships", "url": "https://www.tatatrusts.org/scholarships/rss", "weight": 0.95},
    {"name": "Reliance Foundation", "url": "https://www.reliancefoundation.org/rss/scholarships", "weight": 0.95},
    {"name": "Aditya Birla Scholarships", "url": "https://www.adityabirlafoundation.org/rss", "weight": 0.9},
    {"name": "L'Oréal For Women in Science", "url": "https://www.loreal.in/rss", "weight": 0.9},
    {"name": "Google India Scholarships", "url": "https://buildyourfuture.withgoogle.com/scholarships/rss", "weight": 0.95},
    {"name": "Microsoft India Scholarships", "url": "https://news.microsoft.com/india/feed/", "weight": 0.9},

    # UNIVERSITY & INSTITUTIONS
    {"name": "DU Scholarships", "url": "https://www.du.ac.in/index.php/scholarships/rss", "weight": 0.85},
    {"name": "IIT Delhi Scholarships", "url": "https://ird.iitd.ac.in/scholarships/rss", "weight": 0.85},
    {"name": "IIT Bombay Scholarships", "url": "https://www.iitb.ac.in/en/scholarships/rss", "weight": 0.85},
    {"name": "JNU Scholarships", "url": "https://www.jnu.ac.in/scholarships/rss", "weight": 0.85},

    # AGGREGATORS
    {"name": "Scholarships in India", "url": "https://www.scholarshipsinindia.com/feed/", "weight": 0.95},
    {"name": "Buddy4Study", "url": "https://www.buddy4study.com/blog/feed/", "weight": 0.95},
    {"name": "India Scholarships", "url": "https://www.indiascholarships.org.in/feed/", "weight": 0.9},
]

INTERNSHIP_SOURCES = [
    # TOP TECH COMPANIES
    {"name": "Google Careers India", "url": "https://careers.google.com/jobs/results/?company=Google&location=India&rss", "weight": 1.0},
    {"name": "Microsoft India Careers", "url": "https://careers.microsoft.com/in/en/rss", "weight": 1.0},
    {"name": "Amazon India Careers", "url": "https://www.amazon.jobs/en/teams/india/rss", "weight": 1.0},
    {"name": "Flipkart Careers", "url": "https://www.flipkartcareers.com/feed/", "weight": 0.95},
    {"name": "Swiggy Careers", "url": "https://www.swiggy.com/careers/rss", "weight": 0.95},
    {"name": "Zomato Careers", "url": "https://www.zomato.com/careers/rss", "weight": 0.95},

    # STARTUPS & UNICORNS
    {"name": "Byju's Careers", "url": "https://byjus.com/careers/feed/", "weight": 0.95},
    {"name": "Unacademy Careers", "url": "https://unacademy.com/careers/rss", "weight": 0.95},
    {"name": "Paytm Careers", "url": "https://paytm.com/careers/feed/", "weight": 0.95},
    {"name": "Ola Careers", "url": "https://careers.olacabs.com/feed/", "weight": 0.95},

    # GOVERNMENT & PSU
    {"name": "ISRO Careers", "url": "https://www.isro.gov.in/careers/rss", "weight": 0.95},
    {"name": "DRDO Careers", "url": "https://www.drdo.gov.in/careers/rss", "weight": 0.95},
    {"name": "BARC Careers", "url": "https://www.barc.gov.in/careers/rss", "weight": 0.95},
    {"name": "Sarkari Naukri", "url": "https://www.sarkarinaukri.com/feed/", "weight": 0.95},

    # BANKING & FINANCE
    {"name": "RBI Careers", "url": "https://www.rbi.org.in/Scripts/RSSFeed.aspx", "weight": 0.95},
    {"name": "SEBI Careers", "url": "https://www.sebi.gov.in/rss.html", "weight": 0.95},
    {"name": "ICICI Careers", "url": "https://www.icicicareers.com/rss", "weight": 0.9},
    {"name": "HDFC Careers", "url": "https://www.hdfcbank.com/careers/rss", "weight": 0.9},

    # CONSULTING & FMCG
    {"name": "McKinsey India Careers", "url": "https://www.mckinsey.com/careers/rss", "weight": 0.95},
    {"name": "BCG India Careers", "url": "https://www.bcg.com/careers/rss", "weight": 0.95},
    {"name": "HUL Careers", "url": "https://www.hul.co.in/careers/rss", "weight": 0.95},
    {"name": "ITC Careers", "url": "https://www.itcportal.com/careers/rss", "weight": 0.95},

    # PORTALS
    {"name": "Internshala", "url": "https://internshala.com/blog/feed/", "weight": 1.0},
    {"name": "LetsIntern", "url": "https://www.letsintern.com/blog/feed/", "weight": 0.95},
    {"name": "Twenty19", "url": "https://twenty19.com/blog/feed", "weight": 0.95},
    {"name": "Naukri Campus", "url": "https://www.naukri.com/campus-recruitment-blog/feed", "weight": 0.95},
    {"name": "Freshersworld", "url": "https://www.freshersworld.com/rss/jobs", "weight": 0.95},
]

SOURCES = SCHOLARSHIP_SOURCES + INTERNSHIP_SOURCES

CATEGORIES = [
    "Scholarships",
    "Internships & Jobs"
]

SUBCATEGORIES = {
    "Scholarships": [
        "Engineering & Tech",
        "Medical & Health",
        "Women & Girls",
        "State-wise",
        "Corporate",
        "Government"
    ],
    "Internships & Jobs": [
        "Tech Giants",
        "Startups",
        "Government & PSU",
        "Banking & Finance",
        "Consulting",
        "Design & Media"
    ]
}

TOP_N = 10  # Show more opportunities
DEDUP_THRESHOLD = 0.85

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === AI MODELS ===
logger.info("Loading AI models...")
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("AI models loaded.")
except Exception as e:
    logger.error(f"Failed to load AI models: {e}")
    exit(1)

# === HELPER: Strict India + Student Filter ===
def is_student_related(text):
    text_lower = text.lower()
    
    # MUST contain India + student keyword
    india_keywords = ['india', 'indian', 'delhi', 'mumbai', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune', 'apply in india', 'for indian students']
    student_keywords = ['scholarship', 'intern', 'internship', 'fresher job', 'apply', 'last date', 'notification', 'admit card', 'exam', 'result', 'career', 'placement', 'recruitment', 'govt job', 'sarkari', 'stipend', 'grant', 'award']

    has_india = any(kw in text_lower for kw in india_keywords)
    has_student = any(kw in text_lower for kw in student_keywords)

    # REJECT global content
    reject_keywords = ['ireland', 'australia', 'uk', 'usa', 'canada', 'europe', 'global', 'international student', 'the pie news', 'tertiary education commission', 'south east technological university']
    has_reject = any(kw in text_lower for kw in reject_keywords)

    return has_india and has_student and not has_reject

# === HELPER: Editorial Dimensions ===
def is_trending(text):
    return any(kw in text.lower() for kw in ['viral', 'trending', 'breaking', 'record', 'skyrockets'])

def is_must_know(text):
    return any(kw in text.lower() for kw in ['last date', 'apply now', 'urgent', 'deadline', 'closing soon'])

# === FETCH & PARSE ARTICLES ===
def fetch_articles():
    articles = []
    now = datetime.utcnow()
    cutoff = now - timedelta(days=7)

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

                title = getattr(entry, 'title', '').strip()
                if not title:
                    continue

                summary = getattr(entry, 'summary', '')
                if not summary and hasattr(entry, 'content'):
                    try:
                        summary = BeautifulSoup(entry.content[0].value, "html.parser").get_text()
                    except:
                        pass

                summary = summary.replace('\n', ' ').strip()
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                if not summary:
                    summary = "Click to view full details and apply."

                if len(summary) < 20:
                    continue

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
                text_lower = article["text_for_ai"].lower()
                source = article["source"].lower()

                # FORCE classification
                if any(kw in source or kw in text_lower for kw in ['scholarship', 'vidyalakshmi', 'aicte', 'ugc', 'tata', 'reliance', 'google scholarship']):
                    article["category"] = "Scholarships"
                elif any(kw in source or kw in text_lower for kw in ['intern', 'internship', 'google careers', 'microsoft india', 'amazon india', 'flipkart', 'isro', 'sarkari']):
                    article["category"] = "Internships & Jobs"
                else:
                    result = classifier(article["text_for_ai"], CATEGORIES, multi_label=False)
                    article["category"] = result["labels"][0]

                article["category_confidence"] = 1.0

                # Tag
                article["is_trending"] = is_trending(article["text_for_ai"])
                article["is_must_know"] = is_must_know(article["text_for_ai"])

                filtered_articles.append(article)
            else:
                continue
        except Exception as e:
            logger.error(f"Classification failed for '{article['title']}': {e}")
            continue
    return filtered_articles

# === CALCULATE SCORE ===
def calculate_score(article, now):
    score = article["source_weight"]
    age_hours = (now - article["published_at"]).total_seconds() / 3600
    recency_multiplier = max(0.1, 1 - (age_hours / (7*24)))
    score *= recency_multiplier
    score *= article["category_confidence"]
    if article.get("is_must_know", False):
        score *= 1.5
    article["score"] = score
    return score

# === DEDUPLICATE & RANK ===
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
    return [a for i, a in enumerate(articles) if i not in to_remove]

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
    <title>StudentPulse — Scholarships & Internships for Indian Students</title>
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4cc9f0;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: var(--dark);
            min-height: 100vh;
            padding: 0;
            overflow-x: hidden;
        }

        /* === SIDEBAR === */
        .sidebar {
            position: fixed;
            top: 0;
            left: -300px;
            width: 280px;
            height: 100vh;
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            box-shadow: 2px 0 20px rgba(0,0,0,0.1);
            z-index: 1000;
            transition: left 0.3s ease;
            padding: 20px;
            overflow-y: auto;
        }

        .sidebar.active {
            left: 0;
        }

        .menu-toggle {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1001;
            background: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid var(--primary);
            margin-bottom: 20px;
        }

        .sidebar h2 {
            color: var(--primary);
            font-size: 1.5rem;
        }

        .sidebar-section {
            margin-bottom: 30px;
        }

        .sidebar-section h3 {
            color: var(--secondary);
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }

        .sidebar-section a {
            display: block;
            padding: 10px 15px;
            color: var(--dark);
            text-decoration: none;
            border-radius: 8px;
            margin-bottom: 5px;
            transition: all 0.2s;
            font-weight: 500;
        }

        .sidebar-section a:hover {
            background: var(--primary);
            color: white;
            transform: translateX(5px);
        }

        /* === MAIN CONTENT === */
        .main-content {
            padding: 80px 20px 40px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .hero {
            text-align: center;
            margin-bottom: 50px;
            padding: 40px 20px;
            background: rgba(255,255,255,0.9);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .hero h1 {
            font-size: 2.8rem;
            color: var(--primary);
            margin-bottom: 10px;
            font-weight: 800;
        }

        .hero p {
            font-size: 1.2rem;
            color: var(--secondary);
            max-width: 800px;
            margin: 0 auto;
        }

        .date {
            margin-top: 20px;
            color: #666;
            font-weight: 500;
        }

        /* === SECTION === */
        .section {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            backdrop-filter: blur(10px);
        }

        .section-header {
            display: flex;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid var(--primary);
        }

        .section-header h2 {
            font-size: 2.2rem;
            color: var(--primary);
            margin: 0;
        }

        .section-header .icon {
            font-size: 2rem;
            margin-right: 15px;
            color: var(--accent);
        }

        /* === CARD === */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
        }

        .card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }

        .card-image {
            height: 200px;
            background: #f0f0f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-weight: bold;
        }

        .card-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .card-content {
            padding: 20px;
        }

        .card-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin: 0 0 10px;
            color: var(--dark);
            line-height: 1.3;
        }

        .card-summary {
            color: #555;
            margin: 10px 0;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .card-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0 10px;
            font-size: 0.85rem;
            color: #777;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: white;
        }

        .badge-deadline {
            background: var(--danger);
        }

        .badge-popular {
            background: var(--accent);
        }

        .apply-btn {
            display: block;
            width: 100%;
            padding: 12px;
            background: var(--primary);
            color: white;
            text-align: center;
            text-decoration: none;
            font-weight: 600;
            border-radius: 8px;
            transition: background 0.2s;
            margin-top: 10px;
            font-size: 1rem;
        }

        .apply-btn:hover {
            background: var(--secondary);
        }

        /* === FOOTER === */
        footer {
            text-align: center;
            padding: 30px 20px;
            color: white;
            font-size: 0.95rem;
            margin-top: 20px;
        }

        /* === MOBILE === */
        @media (max-width: 768px) {
            .main-content {
                padding: 70px 15px 30px;
            }
            .hero h1 {
                font-size: 2.2rem;
            }
            .card-grid {
                grid-template-columns: 1fr;
            }
            .section-header h2 {
                font-size: 1.8rem;
            }
        }

        /* === SCROLLBAR === */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <!-- SIDEBAR -->
    <button class="menu-toggle" onclick="toggleSidebar()">☰</button>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h2>StudentPulse</h2>
        </div>
        <div class="sidebar-section">
            <h3>🎯 Scholarships</h3>
            {% for subcat in subcategories['Scholarships'] %}
            <a href="#scholarships-{{ loop.index }}">{{ subcat }}</a>
            {% endfor %}
        </div>
        <div class="sidebar-section">
            <h3>💼 Internships & Jobs</h3>
            {% for subcat in subcategories['Internships & Jobs'] %}
            <a href="#internships-{{ loop.index }}">{{ subcat }}</a>
            {% endfor %}
        </div>
    </div>

    <!-- MAIN CONTENT -->
    <div class="main-content">
        <div class="hero">
            <h1>StudentPulse</h1>
            <p>Scholarships & Internships for Indian Students. Direct Apply Links. Zero Fluff.</p>
            <div class="date">{{ now.strftime('%A, %B %d, %Y') }}</div>
        </div>

        <!-- SCHOLARSHIPS -->
        <div class="section" id="scholarships">
            <div class="section-header">
                <span class="icon">🎓</span>
                <h2>Scholarships</h2>
            </div>
            {% if categorized.get('Scholarships') and categorized['Scholarships']|length > 0 %}
            <div class="card-grid">
                {% for article in categorized['Scholarships'] %}
                <div class="card">
                    {% if article.image_url %}
                    <div class="card-image">
                        <img src="{{ article.image_url }}" alt="{{ article.title }}">
                    </div>
                    {% else %}
                    <div class="card-image">Scholarship Opportunity</div>
                    {% endif %}
                    <div class="card-content">
                        <h3 class="card-title">{{ article.title }}</h3>
                        <p class="card-summary">{{ article.summary }}</p>
                        <div class="card-meta">
                            <span>{{ article.source }}</span>
                            {% if article.is_must_know %}
                            <span class="badge badge-deadline">Deadline Near</span>
                            {% elif article.is_trending %}
                            <span class="badge badge-popular">Popular</span>
                            {% endif %}
                        </div>
                        <a href="{{ article.url }}" target="_blank" class="apply-btn">🚀 Apply Now</a>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="text-align:center; padding:40px; color:#666;">No scholarships available right now. Check back soon!</p>
            {% endif %}
        </div>

        <!-- INTERNSHIPS -->
        <div class="section" id="internships">
            <div class="section-header">
                <span class="icon">💼</span>
                <h2>Internships & Jobs</h2>
            </div>
            {% if categorized.get('Internships & Jobs') and categorized['Internships & Jobs']|length > 0 %}
            <div class="card-grid">
                {% for article in categorized['Internships & Jobs'] %}
                <div class="card">
                    {% if article.image_url %}
                    <div class="card-image">
                        <img src="{{ article.image_url }}" alt="{{ article.title }}">
                    </div>
                    {% else %}
                    <div class="card-image">Internship Opportunity</div>
                    {% endif %}
                    <div class="card-content">
                        <h3 class="card-title">{{ article.title }}</h3>
                        <p class="card-summary">{{ article.summary }}</p>
                        <div class="card-meta">
                            <span>{{ article.source }}</span>
                            {% if article.is_must_know %}
                            <span class="badge badge-deadline">Deadline Near</span>
                            {% elif article.is_trending %}
                            <span class="badge badge-popular">Popular</span>
                            {% endif %}
                        </div>
                        <a href="{{ article.url }}" target="_blank" class="apply-btn">🚀 Apply Now</a>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="text-align:center; padding:40px; color:#666;">No internships available right now. Check back soon!</p>
            {% endif %}
        </div>
    </div>

    <footer>
        <p>Curated with ❤️ for Indian students • Updated: {{ now.strftime('%A, %B %d, %Y') }}</p>
    </footer>

    <script>
        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('active');
        }

        // Close sidebar when clicking outside
        document.addEventListener('click', function(event) {
            const sidebar = document.getElementById('sidebar');
            const menuToggle = document.querySelector('.menu-toggle');
            if (!sidebar.contains(event.target) && !menuToggle.contains(event.target)) {
                sidebar.classList.remove('active');
            }
        });
    </script>
</body>
</html>
"""

def generate_html(categorized):
    template = Template(HTML_TEMPLATE)
    html = template.render(
        categorized=categorized,
        subcategories=SUBCATEGORIES,
        now=datetime.utcnow()
    )
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML generated: index.html")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    logger.info("Starting StudentPulse...")
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
    logger.info("✅ Done. StudentPulse updated.")
