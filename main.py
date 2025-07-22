import sqlite3
import uvicorn
import logging
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field

DATABASE_URL = "reviews.db"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger: logging.Logger = logging.getLogger(__name__)

app = FastAPI(
    title="API настроений отзывов",
    description="Анализ тональности отзывов",
)

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Отзыв")


class ReviewResponse(BaseModel):
    id: int
    text: str
    sentiment: Sentiment
    created_at: str


class StatsResponse(BaseModel):
    total_reviews: int
    positive: int
    negative: int
    neutral: int


def analyze_sentiment(text: str) -> Sentiment:
    text_lower = text.lower().strip()

    if not text_lower:
        return Sentiment.NEUTRAL

    positive = {'хорош', 'люблю', 'супер', 'круто', 'клас', 'прекрасно', 'нравится', 'лучший', 'здорово'}
    negative = {'плохо', 'ужасно', 'отврат', 'ненавиж', 'провал', 'разочарование'}

    pos_score = sum(1 for word in positive if word in text_lower)
    neg_score = sum(1 for word in negative if word in text_lower)

    if pos_score > neg_score:
        return Sentiment.POSITIVE
    if neg_score > pos_score:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


@contextmanager
def _get_db_cursor() -> Generator[sqlite3.Connection, None, None]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail="Внутренняя ошибка базы данных")
    finally:
        if conn:
            conn.close()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
    finally:
        if conn:
            conn.close()


def init_db() -> None:
    with _get_db_cursor() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        logger.info("Установлено соединение с БД")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Запуск приложения...")
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "API настроений отзывов. Документация доступна по /docs"}


@app.post("/reviews", status_code=201)
def create_review(
    review: ReviewCreate,
    db: sqlite3.Connection = Depends(get_db)
) -> ReviewResponse:
    """Добавляет отзыв, определяет тональность и сохраняет в БД"""
    text = review.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Текст отзыва слишком короткий.")

    sentiment = analyze_sentiment(text)
    created_at = datetime.utcnow().isoformat()

    cursor: sqlite3.Cursor = db.cursor()
    cursor.execute(
        "INSERT INTO reviews (text, sentiment, created_at) VALUES (?, ?, ?)",
        (text, sentiment.value, created_at)
    )
    review_id = cursor.lastrowid
    db.commit()

    logger.info(f"Создан отзыв ID={review_id}, тональность: {sentiment}")

    return ReviewResponse(
        id=review_id,
        text=text,
        sentiment=sentiment,
        created_at=created_at
    )


@app.get("/reviews")
def get_reviews(
    sentiment: Optional[Sentiment] = Query(None, description="Фильтр по тональности"),
    limit: int = Query(10, ge=1, le=100),
    db: sqlite3.Connection = Depends(get_db)
) -> List[ReviewResponse]:
    """Получение списка отзывов"""
    query = "SELECT id, text, sentiment, created_at FROM reviews"
    params = []

    if sentiment:
        query += " WHERE sentiment = ?"
        params.append(sentiment.value)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor = db.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()

    return [
        ReviewResponse(
            id=row["id"],
            text=row["text"],
            sentiment=row["sentiment"],
            created_at=row["created_at"]
        )
        for row in rows
    ]


@app.get("/reviews/{review_id}")
def get_review(
    review_id: int,
    db: sqlite3.Connection = Depends(get_db)
) -> ReviewResponse:
    """Получение одного отзыва"""
    row = db.execute(
        "SELECT id, text, sentiment, created_at FROM reviews WHERE id = ?",
        (review_id,)
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Отзыв не найден")

    return ReviewResponse(
        id=row["id"],
        text=row["text"],
        sentiment=row["sentiment"],
        created_at=row["created_at"]
    )


@app.get("/stats")
def get_stats(db: sqlite3.Connection = Depends(get_db)) -> StatsResponse:
    """Статистика по тональности"""
    cursor = db.cursor()
    cursor.execute("SELECT sentiment, COUNT(*) as count FROM reviews GROUP BY sentiment")
    rows = cursor.fetchall()

    stats = {
        "total_reviews": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    }

    for row in rows:
        key = row["sentiment"]
        if key in stats:
            stats[key] = row["count"]
            stats["total_reviews"] += row["count"]

    return StatsResponse(**stats)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)