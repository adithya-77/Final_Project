import os
import logging
import time
import json
import re
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

class ConversationalDatasetGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.logger.warning("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.target_count = 5  # Default conversations to generate per chunk
    
    def set_target_count(self, count):
        """Set target number of conversations to generate per chunk"""
        self.target_count = count
    
    def generate(self, cleaned_text):
        """Generate conversational dataset from text using Groq API"""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        try:
            # Construct a prompt with explicit instructions for JSON format
            prompt = f"""
            Here is some information:
            {cleaned_text}

            Create a conversational dataset with {self.target_count} realistic conversations based on the information above.
            Each conversation should include multiple turns between a user and assistant discussing topics from the text.
            
            IMPORTANT: You must respond ONLY with a valid JSON array in the following format with no additional text:
            [
                {{
                    "conversation_id": "1",
                    "turns": [
                        {{"role": "user", "content": "First user message"}},
                        {{"role": "assistant", "content": "First assistant response"}},
                        {{"role": "user", "content": "Second user message"}},
                        {{"role": "assistant", "content": "Second assistant response"}}
                    ]
                }},
                {{
                    "conversation_id": "2",
                    "turns": [
                        {{"role": "user", "content": "..."}},
                        {{"role": "assistant", "content": "..."}}
                    ]
                }}
            ]
            """
            
            self.logger.info(f"Sending request to Groq API for conversational dataset generation with target of {self.target_count} conversations")
            
            # Add retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an AI assistant that creates high-quality conversational datasets. Always respond with properly formatted JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama3-8b-8192",
                        temperature=0.8,  # Higher temperature for more diverse conversations
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
                conversations = json.loads(cleaned_response)
                
                # Validate the structure
                if not isinstance(conversations, list):
                    raise ValueError("Response is not a JSON array")
                
                # Flatten the conversation turns into simple user/assistant pairs
                flattened_conversations = []
                for convo in conversations:
                    if "turns" not in convo or not isinstance(convo["turns"], list):
                        continue
                    
                    turns = convo["turns"]
                    for i in range(0, len(turns) - 1, 2):
                        if i + 1 < len(turns):
                            if turns[i]["role"] == "user" and turns[i+1]["role"] == "assistant":
                                flattened_conversations.append({
                                    "user": turns[i]["content"],
                                    "assistant": turns[i+1]["content"]
                                })
                
                return flattened_conversations
                        
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Standard JSON parsing failed: {str(e)}, attempting to extract JSON subset")
                
                # More aggressive extraction - find anything that looks like a JSON array
                json_pattern = r'\[\s*\{.*\}\s*\]'
                match = re.search(json_pattern, cleaned_response, re.DOTALL)
                
                if match:
                    try:
                        potential_json = match.group(0)
                        conversations = json.loads(potential_json)
                        
                        # Flatten the conversation turns
                        flattened_conversations = []
                        for convo in conversations:
                            if "turns" not in convo or not isinstance(convo["turns"], list):
                                continue
                            
                            turns = convo["turns"]
                            for i in range(0, len(turns) - 1, 2):
                                if i + 1 < len(turns):
                                    if turns[i]["role"] == "user" and turns[i+1]["role"] == "assistant":
                                        flattened_conversations.append({
                                            "user": turns[i]["content"],
                                            "assistant": turns[i+1]["content"]
                                        })
                        
                        return flattened_conversations
                    except (json.JSONDecodeError, KeyError) as e:
                        # Last resort fallback: extract user/assistant pairs directly
                        user_pattern = r'"role":\s*"user",\s*"content":\s*"([^"]*)"'
                        assistant_pattern = r'"role":\s*"assistant",\s*"content":\s*"([^"]*)"'
                        
                        user_messages = re.findall(user_pattern, cleaned_response)
                        assistant_messages = re.findall(assistant_pattern, cleaned_response)
                        
                        pairs = []
                        for i in range(min(len(user_messages), len(assistant_messages))):
                            pairs.append({
                                "user": user_messages[i],
                                "assistant": assistant_messages[i]
                            })
                        
                        if pairs:
                            self.logger.info(f"Extracted {len(pairs)} conversation pairs using regex")
                            return pairs
                        else:
                            raise ValueError("Failed to extract conversational data using all methods")
                else:
                    # Final fallback: look for "User:" and "Assistant:" patterns
                    lines = cleaned_response.split("\n")
                    pairs = []
                    current_user = None
                    
                    for line in lines:
                        if line.startswith("User:") or line.startswith("user:"):
                            current_user = line.split(":", 1)[1].strip()
                        elif (line.startswith("Assistant:") or line.startswith("assistant:")) and current_user:
                            assistant = line.split(":", 1)[1].strip()
                            pairs.append({
                                "user": current_user,
                                "assistant": assistant
                            })
                            current_user = None
                    
                    if pairs:
                        self.logger.info(f"Extracted {len(pairs)} conversation pairs using text patterns")
                        return pairs
                    
                    self.logger.error(f"Failed to parse response into conversational data: {cleaned_response[:200]}...")
                    raise ValueError("Could not parse conversational dataset from API response")
            
        except Exception as e:
            self.logger.error(f"Error in conversational dataset generation: {str(e)}")
            raise ValueError(f"API call failed: {str(e)}")