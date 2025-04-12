import pandas as pd
import re
from datetime import datetime
from rouge_score import rouge_scorer

class DatasetEvaluator:
    def __init__(self, csv_file_path, dataset_type):
        self.csv_file_path = csv_file_path
        self.dataset_type = dataset_type.lower()
        self.df = None
        self.issues = []
        self.corrections = []
        self.rouge_scores = {}  # Store ROUGE scores for each row

    def load_data(self):
        """Load the generated CSV file and validate columns"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            if self.dataset_type == "qna":
                required_columns = ['question', 'answer']
            elif self.dataset_type == "conversational":
                required_columns = ['user', 'assistant']
            elif self.dataset_type == "chain-of-thought":
                required_columns = ['Question', 'Reason']
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for {self.dataset_type}: {missing_cols}")
            return True
        except Exception as e:
            self.issues.append(f"Error loading CSV: {e}")
            return False

    def compute_rouge_scores(self, reference_col, hypothesis_col):
        """Compute ROUGE scores between reference and hypothesis columns"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores_dict = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for index, row in self.df.iterrows():
            reference = str(row[reference_col]).strip()
            hypothesis = str(row[hypothesis_col]).strip()
            
            if not reference or not hypothesis:
                self.issues.append(f"Row {index}: Empty {reference_col} or {hypothesis_col} for ROUGE scoring")
                scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            else:
                scores = scorer.score(reference, hypothesis)
                scores = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                }
            
            self.rouge_scores[index] = scores
            for metric in scores:
                scores_dict[metric].append(scores[metric])
        
        # Compute average scores
        avg_scores = {metric: sum(scores) / len(scores) if scores else 0.0 
                     for metric, scores in scores_dict.items()}
        return avg_scores

    def correct_missing_values(self, columns):
        """Correct missing values in specified columns"""
        for col in columns:
            for index, row in self.df.iterrows():
                if pd.isna(row[col]) or str(row[col]).strip() == "":
                    self.issues.append(f"Row {index}: Empty {col}")
                    self.df.at[index, col] = f"MISSING_{col.upper()}"
                    self.corrections.append(f"Row {index}: Added placeholder for empty {col}")

    def correct_question_format(self, question_col):
        """Correct question format issues"""
        for index, row in self.df.iterrows():
            if not isinstance(row[question_col], str):
                continue
            question = str(row[question_col]).strip()
            
            if not question.endswith('?'):
                self.issues.append(f"Row {index}: {question_col} doesn't end with '?'")
                question += '?'
                self.df.at[index, question_col] = question
                self.corrections.append(f"Row {index}: Added missing '?' to {question_col}")
            
            if not question[0].isupper():
                self.issues.append(f"Row {index}: {question_col} not capitalized")
                question = question[0].upper() + question[1:]
                self.df.at[index, question_col] = question
                self.corrections.append(f"Row {index}: Capitalized {question_col}")

    def correct_answer_consistency(self, answer_col):
        """Correct inconsistent answer formats"""
        date_pattern = r'(\d{1,2})[ -]([A-Za-z]+)[ -](\d{4})'
        
        for index, row in self.df.iterrows():
            if not isinstance(row[answer_col], str):
                continue
            answer = str(row[answer_col])
            
            date_match = re.search(date_pattern, answer)
            if date_match:
                try:
                    day, month, year = date_match.groups()
                    standardized_date = f"{int(day)} {month.capitalize()} {year}"
                    if standardized_date != date_match.group():
                        self.issues.append(f"Row {index}: Inconsistent date format in {answer_col}")
                        self.df.at[index, answer_col] = answer.replace(date_match.group(), standardized_date)
                        self.corrections.append(f"Row {index}: Standardized date to {standardized_date}")
                    datetime.strptime(standardized_date, '%d %B %Y')
                except ValueError:
                    self.issues.append(f"Row {index}: Invalid date format in {answer_col}")
                    self.df.at[index, answer_col] = f"INVALID_DATE ({answer})"
                    self.corrections.append(f"Row {index}: Marked invalid date")
            
            if re.search(r'\d+', answer) and 'runs' in answer.lower():
                runs_match = re.search(r'(\d+)\s*(runs)', answer, re.IGNORECASE)
                if runs_match and not re.search(r'\d+ runs', answer):
                    self.issues.append(f"Row {index}: Inconsistent runs format in {answer_col}")
                    new_format = f"{runs_match.group(1)} runs"
                    self.df.at[index, answer_col] = re.sub(r'\d+\s*runs', new_format, answer, flags=re.IGNORECASE)
                    self.corrections.append(f"Row {index}: Standardized to '{new_format}'")

    def handle_duplicates(self, key_col):
        """Remove duplicate entries based on key column"""
        duplicates = self.df.duplicated([key_col], keep='first')
        for index in self.df[duplicates].index:
            self.issues.append(f"Row {index}: Duplicate {key_col} found")
            self.df = self.df.drop(index)
            self.corrections.append(f"Row {index}: Removed duplicate {key_col}")

    def correct_data_types(self, question_col, answer_col):
        """Correct data type inconsistencies"""
        for index, row in self.df.iterrows():
            if not isinstance(row[answer_col], str):
                continue
            question = str(row[question_col]).lower()
            answer = str(row[answer_col])
            
            if 'how many' in question and not any(char.isdigit() for char in answer) and 'most' not in answer:
                self.issues.append(f"Row {index}: Expected numerical {answer_col}")
                self.df.at[index, answer_col] = f"UNKNOWN_NUMBER ({answer})"
                self.corrections.append(f"Row {index}: Added placeholder for missing number")
            
            if 'when' in question and not re.search(r'\d{4}', answer):
                self.issues.append(f"Row {index}: Expected date in {answer_col}")
                self.df.at[index, answer_col] = f"UNKNOWN_DATE ({answer})"
                self.corrections.append(f"Row {index}: Added placeholder for missing date")

    def evaluate_qna(self):
        """Evaluate and correct QnA dataset"""
        self.correct_missing_values(['question', 'answer'])
        self.correct_question_format('question')
        self.correct_answer_consistency('answer')
        self.handle_duplicates('question')
        self.correct_data_types('question', 'answer')
        avg_scores = self.compute_rouge_scores('question', 'answer')
        return avg_scores

    def evaluate_conversational(self):
        """Evaluate and correct Conversational dataset"""
        self.correct_missing_values(['user', 'assistant'])
        
        for index, row in self.df.iterrows():
            user_text = str(row['user']).strip()
            assistant_text = str(row['assistant']).strip()
            
            if not user_text.endswith(('?', '.', '!')):
                self.issues.append(f"Row {index}: User message lacks proper punctuation")
                if '?' in user_text:
                    self.df.at[index, 'user'] = user_text + '?'
                else:
                    self.df.at[index, 'user'] = user_text + '.'
                self.corrections.append(f"Row {index}: Added punctuation to user message")
            
            if len(assistant_text.split()) < 5 and not assistant_text.endswith('?'):
                self.issues.append(f"Row {index}: Assistant response too short")
                self.df.at[index, 'assistant'] = f"{assistant_text} (Elaborated response needed)"
                self.corrections.append(f"Row {index}: Flagged short assistant response")
            
            if index > 0:
                prev_assistant = str(self.df.at[index-1, 'assistant']).lower()
                if 'yes' in assistant_text.lower() and 'no' in prev_assistant:
                    self.issues.append(f"Row {index}: Possible contradiction in conversation flow")
                    self.corrections.append(f"Row {index}: Flagged potential contradiction")
        
        avg_scores = self.compute_rouge_scores('user', 'assistant')
        return avg_scores

    def evaluate_chain_of_thought(self):
        """Evaluate and correct Chain-of-Thought dataset"""
        self.correct_missing_values(['Question', 'Reason'])
        self.correct_question_format('Question')
        
        for index, row in self.df.iterrows():
            question = str(row['Question']).strip()
            reason = str(row['Reason']).strip()
            
            if len(reason.split()) < 10:
                self.issues.append(f"Row {index}: Reason too brief")
                self.df.at[index, 'Reason'] = f"{reason} (More detailed reasoning required)"
                self.corrections.append(f"Row {index}: Flagged brief reasoning")
            
            if "text" not in reason.lower():
                self.issues.append(f"Row {index}: Reason may not reference source text")
                self.df.at[index, 'Reason'] = f"{reason} (Verify text reference)"
                self.corrections.append(f"Row {index}: Flagged potential missing text reference")
            
            if not reason.startswith("Reason:"):
                self.issues.append(f"Row {index}: Reason format inconsistent")
                self.df.at[index, 'Reason'] = f"Reason: {reason}"
                self.corrections.append(f"Row {index}: Standardized reason format")
            
            question_lower = question.lower()
            reason_lower = reason.lower()
            if 'when' in question_lower and not re.search(r'\d{4}', reason_lower):
                self.issues.append(f"Row {index}: Reason lacks expected date for 'when' question")
                self.df.at[index, 'Reason'] = f"{reason} (Expected date missing)"
                self.corrections.append(f"Row {index}: Flagged missing date in reasoning")
            elif 'who' in question_lower and not any(word[0].isupper() for word in reason.split() if len(word) > 2):
                self.issues.append(f"Row {index}: Reason lacks expected proper noun for 'who' question")
                self.df.at[index, 'Reason'] = f"{reason} (Expected name missing)"
                self.corrections.append(f"Row {index}: Flagged missing name in reasoning")
        
        avg_scores = self.compute_rouge_scores('Question', 'Reason')
        return avg_scores

    def run_evaluation(self):
        """Run evaluation based on dataset type and return corrected CSV path and ROUGE scores"""
        if not self.load_data():
            return None, None, None, None
        
        if self.dataset_type == "qna":
            avg_rouge_scores = self.evaluate_qna()
        elif self.dataset_type == "conversational":
            avg_rouge_scores = self.evaluate_conversational()
        elif self.dataset_type == "chain-of-thought":
            avg_rouge_scores = self.evaluate_chain_of_thought()
        
        corrected_path = self.csv_file_path.replace('.csv', '_corrected.csv')
        self.df.to_csv(corrected_path, index=False)
        return corrected_path, self.issues, self.corrections, avg_rouge_scores