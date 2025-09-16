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
import re
import logging

# === CONFIG ===
SCHOLARSHIP_SOURCES = [
    {"name": "National Scholarship Portal", "url": "https://scholarships.gov.in/NewsRSS", "weight": 1.0},
    {"name": "Vidya Lakshmi Portal", "url": "https://www.vidyalakshmi.co.in/NewsFeed", "weight": 0.95},
    {"name": "AICTE Scholarships", "url": "https://www.aicte-india.org/scholarships/rss", "weight": 0.95},
    {"name": "UGC Scholarships", "url": "https://nsp.ugc.ac.in/RSSFeed.aspx", "weight": 0.95},
    {"name": "Tata Trust Scholarships", "url": "https://www.tatatrusts.org/scholarships/rss", "weight": 0.95},
    {"name": "Google India Scholarships", "url": "https://buildyourfuture.withgoogle.com/scholarships/rss", "weight": 0.95},
    {"name": "Buddy4Study", "url": "https://www.buddy4study.com/blog/feed/", "weight": 0.95},
    {"name": "Scholarships in India", "url": "https://www.scholarshipsinindia.com/feed/", "weight": 0.95},
]

INTERNSHIP_SOURCES = [
    {"name": "Google Careers India", "url": "https://careers.google.com/jobs/results/?company=Google&location=India&rss", "weight": 1.0},
    {"name": "Microsoft India Careers", "url": "https://careers.microsoft.com/in/en/rss", "weight": 1.0},
    {"name": "Amazon India Careers", "url": "https://www.amazon.jobs/en/teams/india/rss", "weight": 1.0},
    {"name": "Flipkart Careers", "url": "https://www.flipkartcareers.com/feed/", "weight": 0.95},
    {"name": "Internshala", "url": "https://internshala.com/blog/feed/", "weight": 1.0},
    {"name": "Naukri Campus", "url": "https://www.naukri.com/campus-recruitment-blog/feed", "weight": 0.95},
    {"name": "ISRO Careers", "url": "https://www.isro.gov.in/careers/rss", "weight": 0.95},
    {"name": "Sarkari Naukri", "url": "https://www.sarkarinaukri.com/feed/", "weight": 0.95},
]

SOURCES = SCHOLARSHIP_SOURCES + INTERNSHIP_SOURCES

CATEGORIES = ["Scholarships", "Internships & Jobs"]

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

# === HELPER: Extract Deadline from Text ===
def extract_deadline(text):
    # Look for patterns like "Last date: 30 Sept", "Apply by 5 Oct", etc.
    patterns = [
        r'last\s+date.*?(\d{1,2}\s+[a-zA-Z]+)',
        r'apply\s+by.*?(\d{1,2}\s+[a-zA-Z]+)',
        r'deadline.*?(\d{1,2}\s+[a-zA-Z]+)',
        r'closing\s+on.*?(\d{1,2}\s+[a-zA-Z]+)',
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    return None

# === HELPER: Check if Deadline Passed ===
def is_deadline_passed(deadline_str):
    if not deadline_str:
        return False
    try:
        # Assume current year
        deadline_str_full = f"{deadline_str} {datetime.now().year}"
        deadline = datetime.strptime(deadline_str_full, "%d %b %Y")
        return datetime.now() > deadline
    except:
        return False

# === HELPER: Strict India + Student Filter ===
def is_student_related(text):
    text_lower = text.lower()
    india_keywords = ['india', 'indian', 'delhi', 'mumbai', 'apply in india', 'for indian students']
    student_keywords = ['scholarship', 'intern', 'internship', 'fresher job', 'apply', 'last date', 'deadline', 'notification', 'recruitment', 'govt job', 'sarkari']
    has_india = any(kw in text_lower for kw in india_keywords)
    has_student = any(kw in text_lower for kw in student_keywords)
    reject_keywords = ['ireland', 'australia', 'uk', 'usa', 'canada', 'europe', 'global', 'international student', 'the pie news']
    has_reject = any(kw in text_lower for kw in reject_keywords)
    return has_india and has_student and not has_reject

# === FETCH & PARSE ARTICLES (NO TIME CUTOFF) ===
def fetch_articles():
    articles = []
    now = datetime.utcnow()

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

                # ✅ NO TIME CUTOFF — fetch everything
                # Deadline will be checked separately

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

                # Extract deadline
                deadline_str = extract_deadline(title + " " + summary)
                is_expired = is_deadline_passed(deadline_str)

                if is_expired:
                    continue  # ✅ Skip if deadline passed

                articles.append({
                    "title": title,
                    "summary": summary,
                    "url": entry.link,
                    "source": source["name"],
                    "source_weight": source["weight"],
                    "published_at": pub_date,
                    "text_for_ai": f"{title}. {summary}",
                    "image_url": image_url,
                    "deadline": deadline_str
                })
        except Exception as e:
            logger.error(f"Error fetching {source['name']}: {e}")

    logger.info(f"Fetched {len(articles)} articles (after deadline filter).")
    return articles

# === CLASSIFY ARTICLES ===
def classify_articles(articles):
    filtered_articles = []
    for article in articles:
        try:
            if is_student_related(article["text_for_ai"]):
                text_lower = article["text_for_ai"].lower()
                if any(kw in text_lower for kw in ['scholarship', 'vidyalakshmi', 'aicte', 'ugc', 'tata']):
                    article["category"] = "Scholarships"
                elif any(kw in text_lower for kw in ['intern', 'internship', 'google careers', 'microsoft india', 'amazon india', 'isro', 'sarkari']):
                    article["category"] = "Internships & Jobs"
                else:
                    result = classifier(article["text_for_ai"], CATEGORIES, multi_label=False)
                    article["category"] = result["labels"][0]
                article["category_confidence"] = 1.0
                article["is_must_know"] = bool(article.get("deadline"))
                filtered_articles.append(article)
            else:
                continue
        except Exception as e:
            logger.error(f"Classification failed for '{article['title']}': {e}")
            continue
    return filtered_articles

# === SCORE & DEDUPLICATE ===
def calculate_score(article, now):
    score = article["source_weight"]
    if article.get("is_must_know"):
        score *= 1.5
    article["score"] = score
    return score

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
            if cosine_scores[i][j] > 0.85:
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
        top_articles[cat] = sorted_articles  # ✅ Show ALL, not just TOP_N
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
            --primary: #6A35FF; /* Deep Purple */
            --secondary: #00D1FF; /* Electric Blue */
            --accent: #39FF14; /* Neon Green */
            --dark: #0F0F1A;
            --light: #F0F0FF;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }

        body {
            background: linear-gradient(135deg, var(--dark) 0%, #1A1A2E 100%);
            color: var(--light);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* === SIDEBAR === */
        .sidebar {
            position: fixed;
            top: 0;
            left: -320px;
            width: 300px;
            height: 100vh;
            background: rgba(15, 15, 26, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(106, 53, 255, 0.3);
            z-index: 1000;
            transition: left 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 30px 20px;
            overflow-y: auto;
        }

        .sidebar.active {
            left: 0;
        }

        .menu-toggle {
            position: fixed;
            top: 30px;
            left: 30px;
            z-index: 1001;
            background: rgba(106, 53, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid var(--primary);
            color: var(--primary);
            width: 50px;
            height: 50px;
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            transition: all 0.3s ease;
        }

        .menu-toggle:hover {
            background: rgba(106, 53, 255, 0.3);
            transform: rotate(90deg);
        }

        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 30px;
            border-bottom: 2px solid var(--primary);
        }

        .sidebar h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .sidebar-section {
            margin-bottom: 40px;
        }

        .sidebar-section h3 {
            color: var(--secondary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 209, 255, 0.3);
            font-size: 1.2rem;
            font-weight: 600;
        }

        .sidebar-section a {
            display: block;
            padding: 12px 20px;
            color: var(--light);
            text-decoration: none;
            border-radius: 10px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            border-left: 3px solid transparent;
        }

        .sidebar-section a:hover {
            background: rgba(106, 53, 255, 0.2);
            border-left: 3px solid var(--primary);
            transform: translateX(10px);
            color: var(--primary);
        }

        /* === MAIN CONTENT === */
        .main-content {
            padding: 100px 40px 60px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .hero {
            text-align: center;
            margin-bottom: 80px;
            padding: 60px 40px;
            background: linear-gradient(135deg, rgba(106, 53, 255, 0.1), rgba(0, 209, 255, 0.1));
            border-radius: 30px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(106, 53, 255, 0.3);
            box-shadow: 0 10px 50px rgba(0, 0, 0, 0.3);
        }

        .hero h1 {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(to right, var(--primary), var(--secondary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
            letter-spacing: -1px;
        }

        .hero p {
            font-size: 1.4rem;
            color: var(--secondary);
            max-width: 900px;
            margin: 0 auto 30px;
            line-height: 1.6;
        }

        .date {
            background: rgba(57, 255, 20, 0.1);
            border: 1px solid var(--accent);
            color: var(--accent);
            padding: 8px 20px;
            border-radius: 50px;
            font-weight: 600;
            display: inline-block;
            font-size: 1.1rem;
        }

        /* === SECTION === */
        .section {
            margin-bottom: 80px;
            scroll-margin-top: 100px;
        }

        .section-header {
            display: flex;
            align-items: center;
            margin-bottom: 50px;
        }

        .section-header .icon {
            font-size: 3rem;
            margin-right: 20px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .section-header h2 {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            letter-spacing: -1px;
        }

        /* === CARD GRID === */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 30px;
        }

        .card {
            background: rgba(30, 30, 50, 0.6);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(106, 53, 255, 0.3);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 50px rgba(106, 53, 255, 0.3);
            border-color: var(--primary);
        }

        .card-image {
            height: 220px;
            background: linear-gradient(45deg, #1A1A2E, #16213E);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--primary);
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            padding: 20px;
        }

        .card-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }

        .card:hover .card-image img {
            transform: scale(1.05);
        }

        .card-content {
            padding: 30px;
        }

        .card-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0 0 15px;
            line-height: 1.4;
            color: white;
        }

        .card-summary {
            color: #B0B0D0;
            margin: 15px 0;
            font-size: 1rem;
            line-height: 1.6;
        }

        .card-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0 15px;
        }

        .source {
            color: #8080A0;
            font-size: 0.9rem;
        }

        .badge {
            padding: 6px 16px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .badge-deadline {
            background: linear-gradient(to right, #FF3860, #FF6E7F);
            color: white;
        }

        .apply-btn {
            display: block;
            width: 100%;
            padding: 16px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            color: white;
            text-align: center;
            text-decoration: none;
            font-weight: 700;
            border-radius: 12px;
            transition: all 0.3s ease;
            margin-top: 15px;
            font-size: 1.1rem;
            border: none;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(106, 53, 255, 0.4);
        }

        .apply-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(106, 53, 255, 0.6);
        }

        /* === FOOTER === */
        footer {
            text-align: center;
            padding: 40px 20px;
            color: #8080A0;
            font-size: 1rem;
            margin-top: 40px;
            border-top: 1px solid rgba(106, 53, 255, 0.2);
        }

        /* === MOBILE === */
        @media (max-width: 768px) {
            .main-content {
                padding: 120px 20px 40px;
            }
            .hero {
                padding: 40px 20px;
            }
            .hero h1 {
                font-size: 2.5rem;
            }
            .hero p {
                font-size: 1.1rem;
            }
            .card-grid {
                grid-template-columns: 1fr;
            }
            .section-header h2 {
                font-size: 2.2rem;
            }
            .menu-toggle {
                top: 20px;
                left: 20px;
            }
        }

        /* === SMOOTH SCROLL === */
        html {
            scroll-behavior: smooth;
        }
    </style>
</head>
<body>
    <!-- SIDEBAR -->
    <button class="menu-toggle" onclick="toggleSidebar()">☰</button>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h1>StudentPulse</h1>
        </div>
        <div class="sidebar-section">
            <h3>🎯 Scholarships</h3>
            <a href="#scholarships">All Scholarships</a>
        </div>
        <div class="sidebar-section">
            <h3>💼 Internships & Jobs</h3>
            <a href="#internships">All Internships</a>
        </div>
    </div>

    <!-- MAIN CONTENT -->
    <div class="main-content">
        <div class="hero">
            <h1>StudentPulse</h1>
            <p>Every Scholarship. Every Internship. Zero Fluff. Apply Before Deadline.</p>
            <div class="date">{{ now.strftime('%A, %B %d, %Y') }}</div>
        </div>

        <!-- SCHOLARSHIPS -->
        <div class="section" id="scholarships">
            <div class="section-header">
                <div class="icon">🎓</div>
                <h2>Scholarships</h2>
            </div>
            {% if categorized.get('Scholarships') and categorized['Scholarships']|length > 0 %}
            <div class="card-grid">
                {% for article in categorized['Scholarships'] %}
                <div class="card">
                    <div class="card-image">
                        {% if article.image_url %}
                            <img src="{{ article.image_url }}" alt="{{ article.title }}">
                        {% else %}
                            <img src="https://via.placeholder.com/400x220/6A35FF/FFFFFF?text=Scholarship+Opportunity" alt="Scholarship">
                        {% endif %}
                    </div>
                    <div class="card-content">
                        <h3 class="card-title">{{ article.title }}</h3>
                        <p class="card-summary">{{ article.summary }}</p>
                        <div class="card-meta">
                            <span class="source">{{ article.source }}</span>
                            {% if article.deadline %}
                            <span class="badge badge-deadline">Deadline: {{ article.deadline }}</span>
                            {% endif %}
                        </div>
                        <a href="{{ article.url }}" target="_blank" class="apply-btn">🚀 Apply Now</a>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="text-align:center; padding:60px; color:#8080A0; font-size:1.2rem;">No active scholarships found. Check back soon!</p>
            {% endif %}
        </div>

        <!-- INTERNSHIPS -->
        <div class="section" id="internships">
            <div class="section-header">
                <div class="icon">💼</div>
                <h2>Internships & Jobs</h2>
            </div>
            {% if categorized.get('Internships & Jobs') and categorized['Internships & Jobs']|length > 0 %}
            <div class="card-grid">
                {% for article in categorized['Internships & Jobs'] %}
                <div class="card">
                    <div class="card-image">
                        {% if article.image_url %}
                            <img src="{{ article.image_url }}" alt="{{ article.title }}">
                        {% else %}
                            <img src="https://via.placeholder.com/400x220/00D1FF/FFFFFF?text=Internship+Available" alt="Internship">
                        {% endif %}
                    </div>
                    <div class="card-content">
                        <h3 class="card-title">{{ article.title }}</h3>
                        <p class="card-summary">{{ article.summary }}</p>
                        <div class="card-meta">
                            <span class="source">{{ article.source }}</span>
                            {% if article.deadline %}
                            <span class="badge badge-deadline">Deadline: {{ article.deadline }}</span>
                            {% endif %}
                        </div>
                        <a href="{{ article.url }}" target="_blank" class="apply-btn">🚀 Apply Now</a>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="text-align:center; padding:60px; color:#8080A0; font-size:1.2rem;">No active internships found. Check back soon!</p>
            {% endif %}
        </div>
    </div>

    <footer>
        <p>Curated with ❤️ for Indian students • Updated daily • Missed a deadline? It's automatically removed.</p>
    </footer>

    <script>
        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('active');
        }

        // Close sidebar on outside click
        document.addEventListener('click', function(event) {
            const sidebar = document.getElementById('sidebar');
            const menuToggle = document.querySelector('.menu-toggle');
            if (!sidebar.contains(event.target) && !menuToggle.contains(event.target)) {
                sidebar.classList.remove('active');
            }
        });

        // Smooth scroll (already enabled via CSS: html { scroll-behavior: smooth; })
    </script>
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
