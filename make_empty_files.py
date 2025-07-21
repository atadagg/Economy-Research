import os

NEWS_DIR = "news"
OLD3_DIR = "older/3"

for channel in os.listdir(NEWS_DIR):
    news_channel_path = os.path.join(NEWS_DIR, channel)
    old3_channel_path = os.path.join(OLD3_DIR, channel)
    if not os.path.isdir(news_channel_path):
        continue

    # Ensure the old/3/<channel>/ directory exists
    os.makedirs(old3_channel_path, exist_ok=True)

    for filename in os.listdir(news_channel_path):
        news_file = os.path.join(news_channel_path, filename)
        old3_file = os.path.join(old3_channel_path, filename)
        if not os.path.isfile(news_file):
            continue
        if not os.path.exists(old3_file):
            # Create an empty file
            open(old3_file, "w").close()
            print(f"Created empty: {old3_file}")