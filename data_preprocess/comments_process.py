from rich import print
import random
import sqlite3
from langdetect import detect
from faker import Faker
from tqdm import tqdm
from datetime import datetime

from text_processing import classify_comment_roberta


fake = Faker()

categories = [
    "about the brand",
    "about the product",
    "about the service",
    "about the company",
    "cooperation",
    "competitors",
    "discrimination",
    "spam",
    "extreme profanity",
    "brand attacks",
    "FAQs",
    "customer complaints",
    "fan communities",
    "social",
    "political",
    "business",
    "bullying",
    "miscellaneous",
    "other",
]

tags = [
    "youtube",
    "twitter",
    "facebook",
    "instagram",
    "tiktok",
    "email",
    "website",
    "social media",
    "evergreen",
    "event",
    "ad",
    "education",
    "news",
    "holiday",
    "promotion",
    "brand",
    "company",
    "product",
    "service",
    "published",
    "unpublished",
    "sheduled",
    "unscheduled",
    "aproved",
    "rejected",
    "video",
    "audio",
    "image",
    "art",
    "post",
]


def analyze_comment(
    comment_text: str,
) -> tuple[str, str, str, str, str, str, str, str]:
    """
    This function is used to generate a fake comment analisys and its corresponding sentiment, emotion, language, categories, tags, timestamp, and user ID.
    """
    # sentiment, emotion, language = classify_comment_roberta(comment_text)  # For using roberta model to get sentiment and emotion
    sentiment = random.choice(["awesome", "happy", "neutral", "sad", "angry", "confused"])
    emotion = random.choice(["disappointment", "negative", "joy", "optimism", "surprise", "love", "anger", "sadness", "fear", "confusion", "disgust", "neutral"])
    language = detect(comment_text)
    cats_list = ",".join(random.choices(categories, k=random.randint(1, 5)))
    tags_list = ",".join(random.choices(tags, k=random.randint(1, 5)))
    timestamp = fake.date_between(start_date="-1m", end_date="today")
    user_id = random.randint(1, 20)
    return (
        comment_text,
        sentiment,
        emotion,
        language,
        cats_list,
        tags_list,
        timestamp,
        user_id,
    )


def main():
    with open("/home/wsl/brandbastion/bb_test/comments.txt", "r") as f:
        tmp_text = f.read()

    comments_data = []
    for comment in tqdm(tmp_text.split('","')):
        comments_data.append(analyze_comment(comment))

    with open("/home/wsl/brandbastion/bb_test/comments_data.txt", "w") as f:
        for comment in comments_data:
            f.write(f"{comment}\n")

    users_data = [(fake.user_name(),) for _ in range(20)]

    with open("/home/wsl/brandbastion/bb_test/users_data.txt", "w") as f:
        for user in users_data:
            f.write(f"{user}\n")

    conn = sqlite3.connect("./db/brandbastion.db")
    cursor = conn.cursor()

    cursor.executemany(
        """
        INSERT INTO comments (comment_text, sentiment, emotion, language, categories, tags, timestamp, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        comments_data,
    )

    cursor.executemany(
        """
        INSERT INTO users (username)
        VALUES (?)
    """,
        users_data,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
