import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=TOKEN)
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 500) -> List[str]:
    """Splits large text into manageable chunks with overlap to maintain context."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def get_embeddings_from_huggingface(texts: List[str], 
                                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    """Get embeddings using Hugging Face's API - no local installation needed."""
    embeddings = []
    print(f"Getting embeddings for {len(texts)} text chunks...")
    
    # Process in batches to avoid overwhelming the API
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Get embeddings for a batch of texts
            batch_embeddings = client.feature_extraction(
                model=model_name,
                inputs=batch,
            )
            
            # Add to our list of embeddings
            embeddings.extend(batch_embeddings)
            
            # Simple progress indicator
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
            
            # Brief pause to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error getting embeddings for batch starting at index {i}: {e}")
            # If we fail, return empty embeddings for this batch
            embeddings.extend([[0.0] * 384] * len(batch))  # 384 is the dimension for all-MiniLM-L6-v2
            
    return embeddings

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    # Convert to numpy arrays if they aren't already
    v1 = np.array(embedding1)
    v2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0
        
    return dot_product / (norm1 * norm2)

def compute_subtopic_relevance(subtopic: str, chunk_embeddings: List[List[float]]) -> List[float]:
    """Compute relevance scores of each chunk to a specific subtopic."""
    # Get embedding for the subtopic
    subtopic_embedding = get_embeddings_from_huggingface([subtopic])[0]
    
    # Calculate similarity with each chunk
    relevance_scores = [calculate_similarity(subtopic_embedding, chunk_embedding) 
                        for chunk_embedding in chunk_embeddings]
    
    return relevance_scores

def extract_relevant_chunks(chunks: List[str], relevance_scores: List[float], 
                            threshold: float = 0.3, max_chunks: int = 10) -> List[Tuple[str, float]]:
    """Extract the most relevant chunks based on relevance scores."""
    # Create (chunk, score) pairs
    chunk_scores = list(zip(chunks, relevance_scores))
    
    # Filter by threshold and sort by relevance
    relevant_chunks = [(chunk, score) for chunk, score in chunk_scores if score > threshold]
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top chunks (limited by max_chunks)
    return relevant_chunks[:max_chunks]

def generate_summaries(text: str, topic_structure: Dict[str, List[str]], 
                       output_file: str = "book_summaries.json",
                       relevance_threshold: float = 0.3) -> Dict[str, Dict[str, str]]:
    """Generate detailed summaries for each subtopic from a book-length text using embeddings."""
    summaries = {"topic": topic_structure["topic"], "subtopics": {}}
    
    # Chunk the text
    text_chunks = chunk_text(text)
    total_chunks = len(text_chunks)
    print(f"Processing book with {len(text)} characters in {total_chunks} chunks")
    
    # Generate embeddings for all chunks (only done once)
    print("Generating embeddings for all chunks...")
    chunk_embeddings = get_embeddings_from_huggingface(text_chunks)
    print("Embeddings generated.")
    
    # Process each subtopic
    for subtopic in topic_structure["subtopics"]:
        print(f"Generating summary for subtopic: '{subtopic}'")
        
        # Find chunks relevant to this subtopic
        relevance_scores = compute_subtopic_relevance(subtopic, chunk_embeddings)
        relevant_chunks = extract_relevant_chunks(text_chunks, relevance_scores, 
                                                 threshold=relevance_threshold)
        
        print(f"  Found {len(relevant_chunks)} relevant chunks for '{subtopic}'")
        
        # If no relevant chunks found, continue to next subtopic
        if not relevant_chunks:
            summaries["subtopics"][subtopic] = "No relevant information found in the text."
            continue
            
        full_summary = ""
        
        # Process each relevant chunk
        for i, (chunk, score) in enumerate(relevant_chunks):
            print(f"  Processing relevant chunk {i+1}/{len(relevant_chunks)} (relevance: {score:.4f})")
            
            # Create a prompt that includes the relevance context
            prompt = f"""
            You are summarizing parts of a book. Focus specifically on the subtopic: "{subtopic}".
            This chunk has been identified as having content relevant to this subtopic with a relevance score of {score:.4f}.
            
            Book text chunk:
            {chunk}
            
            Provide a clear, well-structured summary of content related to "{subtopic}" from this chunk only.
            Be precise and focus only on the most relevant information.
            """
            
            try:
                # Add retries for API stability
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.chat_completion(
                            messages=[{"role": "user", "content": prompt}], 
                            model=model_id,
                            temperature=0.3  # Lower temperature for more focused responses
                        )
                        summary_text = response.content.strip()
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"    Retry {attempt+1}/{max_retries} after error: {e}")
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise e
                
                # Add to the full summary with the relevance score
                full_summary += f"[Relevance: {score:.4f}] {summary_text}\n\n"
                
                # Save progress after each chunk to prevent data loss
                summaries["subtopics"][subtopic] = full_summary.strip()
                with open(output_file, 'w') as f:
                    json.dump(summaries, f, indent=2)
                    
                # Avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error generating summary for '{subtopic}', chunk {i+1}: {e}")
                full_summary += f"[Error processing chunk {i+1}]\n\n"
        
        # Final processing to create coherent summary
        if full_summary.strip():
            coherence_prompt = f"""
            Below are extracted summaries about "{subtopic}" from different parts of a book.
            Each summary is prefixed with its relevance score (higher is more relevant).
            
            Please create one coherent, well-structured comprehensive summary that combines all this information.
            Give more weight to the information from higher-relevance sections.
            
            {full_summary}
            """
            
            try:
                response = client.chat_completion(
                    messages=[{"role": "user", "content": coherence_prompt}], 
                    model=model_id
                )
                summaries["subtopics"][subtopic] = response.content.strip()
            except Exception as e:
                print(f"❌ Error generating final coherent summary for '{subtopic}': {e}")
                # Keep the concatenated summaries if final processing fails
        
        # Save after completing each subtopic
        with open(output_file, 'w') as f:
            json.dump(summaries, f, indent=2)
    
    print(f"✅ All summaries generated and saved to {output_file}")
    return summaries

def analyze_topics_in_book(book_text: str, output_file: str = "book_analysis.json") -> Dict:
    """Automatically discover key topics in a book and generate summaries."""
    # Chunk the text
    text_chunks = chunk_text(book_text)
    
    # Generate topic discovery prompt
    sample_chunks = text_chunks[:5] + text_chunks[len(text_chunks)//2:len(text_chunks)//2+3] + text_chunks[-5:]
    sample_text = "\n\n---\n\n".join(sample_chunks)
    
    topic_discovery_prompt = f"""
    Based on the following excerpts from a book, identify:
    1. The main topic or theme of the book
    2. 5-7 important subtopics that would be valuable to analyze
    
    Format your response as valid JSON with this structure:
    {{
        "topic": "Main Book Topic",
        "subtopics": ["Subtopic 1", "Subtopic 2", "Subtopic 3", ...]
    }}
    
    Book excerpts:
    {sample_text}
    
    Return ONLY the JSON object without any additional text or explanations.
    """
    
    try:
        print("Discovering topics from book content...")
        response = client.chat_completion(
            messages=[{"role": "user", "content": topic_discovery_prompt}], 
            model=model_id
        )
        
        # Extract and parse the JSON response
        topic_structure = json.loads(response.content.strip())
        print(f"Discovered main topic: {topic_structure['topic']}")
        print(f"Discovered subtopics: {', '.join(topic_structure['subtopics'])}")
        
        # Generate summaries using the discovered topics
        return generate_summaries(book_text, topic_structure, output_file)
        
    except Exception as e:
        print(f"❌ Error during topic discovery: {e}")
        # Fallback to generic topics
        fallback_structure = {
            "topic": "Book Content Analysis",
            "subtopics": [
                "Main Characters", 
                "Plot Summary", 
                "Key Themes",
                "Setting and Context",
                "Writing Style"
            ]
        }
        print("Using fallback topic structure")
        return generate_summaries(book_text, fallback_structure, output_file)

# Example usage:
"""
with open("my_book.txt", "r", encoding="utf-8") as f:
    book_text = f.read()

# Option 1: Automatically discover topics
result = analyze_topics_in_book(book_text)

# Option 2: Use predefined topics
topic_structure = {
    "topic": "Main Book Theme", 
    "subtopics": [
        "Character Development", 
        "Plot Arc", 
        "Symbolism",
        "Historical Context",
        "Key Themes"
    ]
}
result = generate_summaries(book_text, topic_structure)
"""