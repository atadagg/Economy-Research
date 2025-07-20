import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Turkish economic keywords for filtering
ECONOMIC_KEYWORDS = [
    "ekonomi", "ekonomik", "enflasyon", "asgari ücret", "maaş", "ücret", 
    "yatırım", "borsa", "hisse", "döviz", "kur", "faiz", "merkez bankası",
    "bütçe", "mali", "maliye", "ihracat", "ithalat", "ticaret", "istihdam",
    "işsizlik", "çalışan", "şirket", "piyasa", "fiyat", "kar", "zarar"
]

# Processing settings
SEGMENT_MIN_LENGTH = 100
SEGMENT_MAX_LENGTH = 1000  
ECONOMIC_THRESHOLD = 0.6 