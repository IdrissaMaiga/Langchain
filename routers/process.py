from fastapi import APIRouter, Query
from typing import Optional
from services.quizcontroller import generate_quiz  # Assuming quiz generation logic is in services.quiz_generator

router = APIRouter()

@router.get("/get_quiz/")
async def get_quiz(
    topic: str = Query(..., description="Main topic for the quiz"),
    num_questions_per_subtopic: int = Query(3, description="Number of questions per subtopic"),
    code: str = Query("", description="give a code the task")
):
    """
    Endpoint to generate a quiz based on a given topic.
    """
    quiz = generate_quiz(topic, num_questions_per_subtopic, code)
    return quiz.model_dump()
