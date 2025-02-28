from typing import List
import os
import json
import re
from huggingface_hub import InferenceClient
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from types import SimpleNamespace
from dotenv import load_dotenv
from sumirize import generate_summaries
# Initialize Hugging Face client
load_dotenv()
TOKEN = os.getenv("HF_TOKEN") 
client = InferenceClient(token=TOKEN)
model_id = "mistralai/Mistral-7B-Instruct-v0.3"




def convert_to_dict(namespace_obj):
    """
    Convert a SimpleNamespace object to a general dictionary format.
    
    Args:
    - namespace_obj: SimpleNamespace instance containing 'topic' and 'subtopics'.
    
    Returns:
    - A dictionary with 'topic' and 'subtopics'.
    """
    # Extract topic and subtopics
    topic = getattr(namespace_obj, 'topic', None)
    subtopics = getattr(namespace_obj, 'subtopics', [])
    
    # Create the general dictionary structure
    topic_structure = {
        "topic": topic,
        "subtopics": subtopics
    }
    
    return topic_structure
# --- Define Models ---
class QuestionAnswer(BaseModel):
    answer_text: str = Field(description="Answer option for the question")
    is_correct: bool = Field(description="True if this is the correct answer")
    explanation: str = Field(description="Explanation of why the answer is correct or incorrect")

class QuizQuestion(BaseModel):
    question_text: str = Field(description="The quiz question")
    answers: List[QuestionAnswer] = Field(description="List of possible answers")

class QuizSubtopic(BaseModel):
    subtopic: str = Field(description="The subtopic for the quiz")
    questions: List[QuizQuestion] = Field(description="List of quiz questions")

class QuizTopic(BaseModel):
    topic: str = Field(description="The main topic")
    subtopics: List[QuizSubtopic] = Field(description="List of subtopics with quizzes")

parser = PydanticOutputParser(pydantic_object=QuizTopic)

# --- Step 1: Generate Subtopics ---
generate_subtopics_prompt = PromptTemplate(
    template="""
    List subtopics for the topicdata: {topicdata}.
    Return ONLY a JSON object with the topic name and a list of subtopics, nothing else. For example:
    {{"topic":"Topic Name",
    "subtopics":["subtopic1", "subtopic2", "subtopic3"]}}
    """,
    input_variables=["topicdata"],
)

def generate_subtopics(topicdata: str):
    # Request to generate subtopics using Hugging Face model
    response = client.chat_completion(
        messages=[{"role": "user", "content": generate_subtopics_prompt.format(topicdata=topicdata)}],
        model=model_id
    )
    
    subtopics_text = response.choices[0].message.content.strip()
    
    # Try to extract JSON from the response
    try:
        # First, try to parse as direct JSON
        json_data = json.loads(subtopics_text)
        print(f"🟢 Debug: Parsed JSON directly: {json_data}")
        
        if isinstance(json_data, dict):
            # Add missing fields if necessary
            if "topic" not in json_data or not json_data["topic"]:
                json_data["topic"] = "non"
            if "subtopics" not in json_data or not json_data["subtopics"]:
                json_data["subtopics"] = [f"{json_data.get('topic', topicdata)} Overview"]
                
            return SimpleNamespace(**json_data)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON directly: {e}")
    
    # If direct parsing fails, try to extract JSON using regex
    json_match = re.search(r'\{.*\}', subtopics_text, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(0))
            print(f"🟢 Debug: Parsed JSON via regex: {json_data}")
            
            if isinstance(json_data, dict):
                if "topic" not in json_data:
                    json_data["topic"] = "non"
                if "subtopics" not in json_data or not json_data["subtopics"]:
                    json_data["subtopics"] = [f"{json_data.get('topic', topicdata)} Overview"]
                return SimpleNamespace(**json_data)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON from regex: {e}")
    
    # Fallback if both parsing methods fail
    print("❌ Failed to parse JSON. Returning fallback data.")
    return SimpleNamespace(
        topic=topicdata,
        subtopics=[f"{topicdata} Overview"]
    )
    
# --- Step 2: Generate Questions for Each Subtopic ---
generate_questions_prompt = PromptTemplate(
    template="""
    Generate {num_questions} multiple-choice questions for the subtopic: {subtopic}.
    
    Additional context from user documents: {extradata}
    
    Each question should have exactly 1 correct answer and 3-4 incorrect answers.
    
    Return ONLY valid JSON that follows this exact structure:
    {{
      "questions": [
        {{
          "question_text": "Question here?",
          "answers": [
            {{"answer_text": "Option A", "is_correct": true, "explanation": "Why correct"}},
            {{"answer_text": "Option B", "is_correct": false, "explanation": "Why incorrect"}},
            {{"answer_text": "Option C", "is_correct": false, "explanation": "Why incorrect"}},
            {{"answer_text": "Option D", "is_correct": false, "explanation": "Why incorrect"}}
          ]
        }}
      ]
    }}
    
    Make sure each question has exactly ONE correct answer (is_correct: true).
    """,
    input_variables=["subtopic", "num_questions", "extradata"],
)

def generate_questions(subtopic: str, num_questions: int, extradata: str = ""):
    response = client.chat_completion(
        messages=[
            {"role": "user", "content": generate_questions_prompt.format(
                subtopic=subtopic, 
                num_questions=num_questions,
                extradata=extradata
            )}
        ], 
        model=model_id
    )
    questions_text = response.choices[0].message.content.strip()
    
    # Extract JSON from the response text
    return extract_json_from_text(questions_text)

# --- Step 3: Validate and Reformat Questions ---
validate_answers_prompt = PromptTemplate(
    template="""
    Check and fix the following JSON to ensure each question has EXACTLY ONE correct answer (is_correct: true).
    
    Return ONLY the fixed JSON with no additional text, explanations, or code blocks.
    The JSON must follow this structure:
    
    {{
      "questions": [
        {{
          "question_text": "Question here?",
          "answers": [
            {{"answer_text": "Option A", "is_correct": true, "explanation": "Why correct?..."}},
            {{"answer_text": "Option B", "is_correct": false, "explanation": "Why incorrect?.."}}
          ]
        }}
      ]
    }}
    
    Here is the JSON to fix:
    {questions_json}
    """,
    input_variables=["questions_json"],
)

def validate_questions(questions_json: str):
    # If the input is already valid JSON in a dictionary form, convert to string
    if isinstance(questions_json, dict):
        questions_json = json.dumps(questions_json)
        
    response = client.chat_completion(
        messages=[
            {"role": "user", "content": validate_answers_prompt.format(questions_json=questions_json)}
        ], 
        model=model_id
    )
    validation_text = response.choices[0].message.content.strip()
    
    # Extract JSON from the response text
    return extract_json_from_text(validation_text)

def extract_json_from_text(text):
    """Extract JSON object from text that might contain explanations or code blocks."""
    # First, try to parse the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON in code blocks
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Look for JSON object pattern
    json_pattern = re.search(r'({[\s\S]*})', text)
    if json_pattern:
        try:
            return json.loads(json_pattern.group(1))
        except json.JSONDecodeError:
            pass
    
    # If we can't find valid JSON, print the problematic text and return an empty structure
    print(f"Failed to extract JSON from: {text}")
    return {"questions": []}

# --- Step 4: Combine Everything into a Final Quiz Object ---
def generate_quiz(topicdata: str, num_questions_per_subtopic: int = 2, extradata: str = ""):
    print(f"Generating quiz for topic: {topicdata}")
    
    # Get subtopics
    namespace_obj = generate_subtopics(topicdata)
    topic = namespace_obj.topic
    subtopics = namespace_obj.subtopics
    topic_structure = convert_to_dict(namespace_obj)
    result = generate_summaries("book", topic_structure)

    print(f"Generated {len(subtopics)} subtopics: {subtopics}")
    
    quiz_subtopics = []
    
    for subtopic in subtopics:
        print(f"Generating questions for subtopic: {subtopic}")
        
        # Generate and validate questions - now passing extradata to the function
        raw_questions = generate_questions(subtopic+"on this topic"+topic, num_questions_per_subtopic, extradata)
        print(f"Generated raw questions for {subtopic}")
        
        validated_questions = validate_questions(raw_questions)
        print(f"Validated questions for {subtopic}")
        
        try:
            # Convert to QuizQuestion objects
            questions = []
            for q_data in validated_questions.get("questions", []):
                answers = [
                    QuestionAnswer(
                        answer_text=a.get("answer_text", ""),
                        is_correct=a.get("is_correct", False),
                        explanation=a.get("explanation", "")
                    )
                    for a in q_data.get("answers", [])
                ]
                
                questions.append(
                    QuizQuestion(
                        question_text=q_data.get("question_text", ""),
                        answers=answers
                    )
                )
            
            if questions:  # Only add subtopic if it has questions
                quiz_subtopics.append(
                    QuizSubtopic(subtopic=subtopic, questions=questions)
                )
                print(f"Added subtopic {subtopic} with {len(questions)} questions")
            else:
                print(f"No valid questions for subtopic {subtopic}")
            
        except Exception as e:
            print(f"Error processing subtopic {subtopic}: {str(e)}")
    
    # Use the topicdata parameter for the topic field
    return QuizTopic(topic=topic, subtopics=quiz_subtopics)

# Example Usage
if __name__ == "__main__":
    main_topic = """ Main Topic: 
Subtopics:
Circulatory System
Anatomy of the Heart
Blood Circulation
Blood Vessels and Their Functions
Blood Pressure Regulation
Cardiac Cycle and Conduction System
Hemodynamics
Common Circulatory Diseases
Role of the Circulatory System in Nutrient Transport"""
    # Optional extra data parameter
    extra_data = "Additional information for context from user documents"
    quiz = generate_quiz(main_topic, num_questions_per_subtopic=3, extradata=extra_data)
    
    # Print summary
    print("\nQuiz Generation Summary:")
    print(f"Topic: {quiz.topic}")
    print(f"Number of subtopics: {len(quiz.subtopics)}")
    for idx, subtopic in enumerate(quiz.subtopics, 1):
        print(f"  {idx}. {subtopic.subtopic}: {len(subtopic.questions)} questions")
    
    # Save to file
    with open("quiz.json", "w") as f:
        f.write(quiz.model_dump_json(indent=2))
    print("\nQuiz saved to quiz.json")