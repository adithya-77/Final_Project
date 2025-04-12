import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

class ChainOfThoughtDatasetGenerator:
    def generate(self, cleaned_text):
        # Initialize Groq client with API key from environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Create a Groq client instance
        client = Groq(api_key=api_key)

        try:
            # Create a prompt for generating conversational datasets
            prompt = f"""
            Below is some text:
            {cleaned_text}

            Generate some dataset based on the content above.Provide questions and their reasons to arrive at the answer in the format:
            User: <Questions>
            Assistant: <Reason>
            """
            
            # Call the Groq API's chat completion endpoint
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",  # Use the correct model
            )
            
            # Parse the response from the Groq API
            generated_content = chat_completion.choices[0].message.content

            # Parse the response into a structured dataset
            conversations = []
            lines = generated_content.split("\n")
            user_message, assistant_response = None, None

            for line in lines:
                if line.startswith("User:"):
                    user_message = line.replace("User:", "").strip()
                elif line.startswith("Assistant:"):
                    assistant_response = line.replace("Assistant:", "").strip()
                    if user_message and assistant_response:
                        conversations.append({"Question": user_message, "Reason": assistant_response})
                        user_message, assistant_response = None, None

            return conversations

        except Exception as e:
            raise ValueError(f"API call failed: {str(e)}")
import os
import logging
import time
import json
import re
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

class ChainOfThoughtDatasetGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.logger.warning("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.target_count = 10  # Default pairs to generate per chunk
    
    def set_target_count(self, count):
        """Set target number of items to generate per chunk"""
        self.target_count = count
    
    def generate(self, cleaned_text):
        """Generate Chain of Thought dataset from text using Groq API"""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        try:
            # Construct a prompt with explicit instructions for JSON format
            prompt = f"""
            Here is some information:
            {cleaned_text}

            Create a Chain of Thought dataset with {self.target_count} examples based on the information above.
            Each example should include a question that requires multi-step reasoning, and a detailed chain of thought that leads to the answer.
            
            IMPORTANT: You must respond ONLY with a valid JSON array in the following format with no additional text:
            [
                {{"question": "Question 1 text", "reasoning": "Step 1: First, I need to consider... Step 2: Next, I should... Final step: Therefore, the answer is..."}},
                {{"question": "Question 2 text", "reasoning": "Step 1: ... Step 2: ... Final step: ..."}},
                ...
            ]
            """
            
            self.logger.info(f"Sending request to Groq API for Chain of Thought generation with target of {self.target_count} items")
            
            # Add retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an AI assistant that creates high-quality Chain of Thought examples. Always respond with properly formatted JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama3-8b-8192",
                        temperature=0.7,  # Slightly higher temperature for creative reasoning
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"API call attempt {attempt+1} failed: {str(e)}. Retrying in {2**attempt} seconds...")
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        raise
            
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
                cot_pairs = json.loads(cleaned_response)
                
                # Validate the structure
                if not isinstance(cot_pairs, list):
                    raise ValueError("Response is not a JSON array")
                    
                # Ensure each item has question and reasoning fields
                for item in cot_pairs:
                    if not isinstance(item, dict) or "question" not in item or "reasoning" not in item:
                        raise ValueError("Invalid data format")
                        
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Standard JSON parsing failed: {str(e)}, attempting to extract JSON subset")
                
                # More aggressive extraction - find anything that looks like a JSON array
                json_pattern = r'\[\s*\{.*\}\s*\]'
                match = re.search(json_pattern, cleaned_response, re.DOTALL)
                
                if match:
                    try:
                        potential_json = match.group(0)
                        cot_pairs = json.loads(potential_json)
                    except json.JSONDecodeError:
                        raise ValueError(f"Failed to parse JSON response after extraction attempt: {cleaned_response[:100]}...")
                else:
                    self.logger.error(f"Failed to parse response into JSON: {cleaned_response[:200]}...")
                    raise ValueError("Could not parse Chain of Thought dataset from API response")
            
            self.logger.info(f"Successfully generated {len(cot_pairs)} Chain of Thought examples")
            
            # Add a delay to avoid exceeding the rate limit
            time.sleep(0.5)
            
            return cot_pairs
            
        except Exception as e:
            self.logger.error(f"Error in Chain of Thought generation: {str(e)}")
            raise ValueError(f"API call failed: {str(e)}")