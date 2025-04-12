import os
import logging
from dotenv import load_dotenv
from groq import Groq
import json
import pandas as pd
import re
import time

class QnADatasetGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.logger.warning("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key)
        self.target_count = 15  # Default QnA pairs to generate per chunk
    
    def set_target_count(self, count):
        """Set target number of QnA pairs to generate per chunk"""
        self.target_count = count
    
    def generate(self, cleaned_text):
        """Generate QnA dataset from text using Groq API"""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        try:
            # Construct a prompt with explicit instructions for JSON format
            prompt = f"""
            Here is some information:
            {cleaned_text}

            Extract exactly {self.target_count} meaningful questions and answers based on the information above.
            Create diverse questions covering different aspects of the text.
            
            IMPORTANT: You must respond ONLY with a valid JSON array in the following format with no additional text:
            [
                {{"question": "Question 1 text", "answer": "Answer 1 text"}},
                {{"question": "Question 2 text", "answer": "Answer 2 text"}},
                ...
            ]
            """
            
            self.logger.info(f"Sending request to Groq API for QnA generation with target of {self.target_count} items")
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that creates high-quality question and answer pairs for training datasets. You must always respond with properly formatted JSON only, with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.5,  # Reduced temperature for more consistent formatting
            )
            
            # Extract the content from the response
            generated_content = chat_completion.choices[0].message.content
            
            # Clean up the response to handle markdown code blocks
            cleaned_response = generated_content.strip()
            # Remove markdown code blocks if present
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Try multiple parsing approaches
            try:
                qna_pairs = json.loads(cleaned_response)
                
                # Validate the structure
                if not isinstance(qna_pairs, list):
                    raise ValueError("Response is not a JSON array")
                    
                # Ensure each item has question and answer fields
                for item in qna_pairs:
                    if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                        raise ValueError("Invalid QnA pair format")
                        
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Standard JSON parsing failed: {str(e)}, attempting to extract JSON subset")
                
                # More aggressive extraction - find anything that looks like a JSON array
                json_pattern = r'\[\s*\{.*\}\s*\]'
                match = re.search(json_pattern, cleaned_response, re.DOTALL)
                
                if match:
                    try:
                        potential_json = match.group(0)
                        qna_pairs = json.loads(potential_json)
                    except json.JSONDecodeError:
                        raise ValueError(f"Failed to parse JSON response after extraction attempt: {cleaned_response[:100]}...")
                else:
                    # Last resort: line-by-line parsing
                    self.logger.warning("All JSON parsing failed, falling back to line-by-line parsing")
                    qna_pairs = []
                    lines = cleaned_response.split("\n")
                    current_question = None
                    
                    for line in lines:
                        line = line.strip()
                        if "question" in line.lower() and ":" in line:
                            parts = line.split(":", 1)
                            current_question = parts[1].strip().strip('"\'')
                        elif "answer" in line.lower() and ":" in line and current_question:
                            parts = line.split(":", 1)
                            answer = parts[1].strip().strip('"\'')
                            qna_pairs.append({"question": current_question, "answer": answer})
                            current_question = None
            
            self.logger.info(f"Successfully generated {len(qna_pairs)} QnA pairs")
            
            # If we didn't get enough QnA pairs, log a warning
            if len(qna_pairs) < self.target_count:
                self.logger.warning(f"Generated only {len(qna_pairs)} QnA pairs, which is less than the target {self.target_count}")
            
            return qna_pairs
        
        except Exception as e:
            self.logger.error(f"Error in QnA generation: {str(e)}")
            raise ValueError(f"API call failed: {str(e)}")
        
        # Add a delay to avoid exceeding the rate limit
        time.sleep(1)  # Adjust this value based on your API usage and rate limit