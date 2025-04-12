import os
import requests
import shutil
import time
import logging
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageEnhance

# Import text generation agents and utilities
from agents.pdf_agent import PDFScrapingAgent
from agents.web_agent import WebsiteScrapingAgent
from agents.preprocessing_agent import PreprocessingAgent
from datasets.qna_generator import QnADatasetGenerator
from datasets.chain_generator import ChainOfThoughtDatasetGenerator
from datasets.convo_generator import ConversationalDatasetGenerator
from datasets.evaluator_agent import DatasetEvaluator
from text_chunker import TextChunker
from batch_processor import BatchProcessor
from utils import save_csv, merge_csv_files, save_processing_stats

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API Keys for Image Generation
PEXELS_API_KEY = "LD1WixF1wfQ0P8LKRxsq2cmIf9tA9PuVPf5OWnBKkAKhb8E8DXPeF8YX"
PIXABAY_API_KEY = "33943553-e906809435cf92f75c8034565"

# Output directory
OUTPUT_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load environment variables for text generation
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize text generation agents
pdf_agent = PDFScrapingAgent()
web_agent = WebsiteScrapingAgent()
preprocessing_agent = PreprocessingAgent()
text_chunker = TextChunker()
qna_generator = QnADatasetGenerator()
chain_generator = ChainOfThoughtDatasetGenerator()
convo_generator = ConversationalDatasetGenerator()

# Image fetching functions
def fetch_images_pexels(query, num_images):
    images = []
    per_page = 80
    pages = (num_images // per_page) + 1
    for page in range(1, pages + 1):
        url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}&page={page}"
        headers = {"Authorization": PEXELS_API_KEY}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "photos" in data:
                images.extend([photo['src']['original'] for photo in data['photos']])
            logger.debug(f"Pexels Response: {data}")
        else:
            logger.error(f"Pexels API Error: {response.status_code} - {response.text}")
        time.sleep(1)
        if len(images) >= num_images:
            break
    return images[:num_images]

def fetch_images_pixabay(query, num_images):
    images = []
    per_page = 200
    pages = (num_images // per_page) + 1
    for page in range(1, pages + 1):
        url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={per_page}&page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "hits" in data:
                images.extend([hit['largeImageURL'] for hit in data['hits']])
            logger.debug(f"Pixabay Response: {data}")
        else:
            logger.error(f"Pixabay API Error: {response.status_code} - {response.text}")
        time.sleep(1)
        if len(images) >= num_images:
            break
    return images[:num_images]

def download_image(url, folder, idx):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(folder, f"image_{idx}.jpg")
            with open(file_path, "wb") as file:
                shutil.copyfileobj(response.raw, file)
            return file_path
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
    return None

def augment_image(image_path, folder, idx):
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        augmented_images = []
        for angle in [90, 180, 270]:
            rotated = img.rotate(angle)
            rotated_path = os.path.join(folder, f"image_{idx}_rotated_{angle}.jpg")
            rotated.save(rotated_path, "JPEG")
            augmented_images.append(rotated_path)
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_path = os.path.join(folder, f"image_{idx}_flipped.jpg")
        flipped.save(flipped_path, "JPEG")
        augmented_images.append(flipped_path)
        for crop_factor in [0.8, 0.6, 0.4]:
            width, height = img.size
            left = width * (1 - crop_factor) / 2
            top = height * (1 - crop_factor) / 2
            right = width - left
            bottom = height - top
            cropped = img.crop((left, top, right, bottom)).resize((width, height))
            cropped_path = os.path.join(folder, f"image_{idx}_cropped_{int(crop_factor*100)}.jpg")
            cropped.save(cropped_path, "JPEG")
            augmented_images.append(cropped_path)
        for factor in [0.5, 1.5]:
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(factor)
            bright_path = os.path.join(folder, f"image_{idx}_bright_{factor}.jpg")
            bright_img.save(bright_path, "JPEG")
            augmented_images.append(bright_path)
            enhancer = ImageEnhance.Contrast(img)
            contrast_img = enhancer.enhance(factor)
            contrast_path = os.path.join(folder, f"image_{idx}_contrast_{factor}.jpg")
            contrast_img.save(contrast_path, "JPEG")
            augmented_images.append(contrast_path)
        return augmented_images
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return []

def generate_image_dataset(prompt, num_unique_images):
    dataset_folder = os.path.join(OUTPUT_DIR, prompt.replace(" ", "_"))
    os.makedirs(dataset_folder, exist_ok=True)
    image_urls = fetch_images_pexels(prompt, num_unique_images // 2) + fetch_images_pixabay(prompt, num_unique_images // 2)
    total_images = 0
    for idx, url in enumerate(image_urls):
        image_path = download_image(url, dataset_folder, idx)
        if image_path:
            augmented_paths = augment_image(image_path, dataset_folder, idx)
            total_images += 1 + len(augmented_paths)
    return total_images, dataset_folder

# Streamlit UI
st.title("Synthetic Dataset Generator")
tab1, tab2 = st.tabs(["Text Dataset", "Image Dataset"])

# Text Dataset Tab
with tab1:
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file. Please configure it.")
        logger.error("GROQ_API_KEY not found in environment variables.")
    else:
        logger.info("GROQ_API_KEY loaded successfully.")
        st.sidebar.title("Upload Input or Provide URL")
        with st.sidebar.expander("Advanced Settings"):
            target_dataset_size = st.number_input("Target dataset size", min_value=50, value=250, key="text_target_size")
            max_concurrent = st.number_input("Max concurrent API calls", min_value=1, max_value=10, value=2, key="text_max_concurrent")
            chunk_size = st.number_input("Approximate chunk size (characters)", min_value=1000, value=4000, key="text_chunk_size")
            text_chunker.max_chars = chunk_size
        input_option = st.sidebar.radio("Choose input type", ("PDF File", "Website URL"), key="text_input_option")
        if "raw_text" not in st.session_state:
            st.session_state.raw_text = None
        if "generated_dataset_file" not in st.session_state:
            st.session_state.generated_dataset_file = None
        if "dataset_type" not in st.session_state:
            st.session_state.dataset_type = None
        if "combined_data" not in st.session_state:
            st.session_state.combined_data = None
        if "corrected_file_name" not in st.session_state:
            st.session_state.corrected_file_name = None

        if input_option == "PDF File":
            uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], key="text_pdf_upload")
            if uploaded_file and st.sidebar.button("Extract Text", key="text_extract_pdf"):
                with st.spinner("Extracting text from PDF..."):
                    try:
                        st.session_state.raw_text = pdf_agent.scrape(uploaded_file)
                        st.success(f"Successfully extracted {len(st.session_state.raw_text)} characters from PDF")
                        logger.info(f"Extracted {len(st.session_state.raw_text)} characters from PDF")
                    except Exception as e:
                        st.error(f"Error extracting text from PDF: {str(e)}")
                        logger.error(f"PDF extraction error: {str(e)}")
                        st.session_state.raw_text = None
        elif input_option == "Website URL":
            url = st.sidebar.text_input("Enter Website URL", key="text_url_input")
            if url and st.sidebar.button("Scrape Website", key="text_scrape_web"):
                with st.spinner(f"Scraping content from {url}..."):
                    try:
                        st.session_state.raw_text = web_agent.scrape(url)
                        st.success(f"Successfully scraped {len(st.session_state.raw_text)} characters from {url}")
                        logger.info(f"Scraped {len(st.session_state.raw_text)} characters from {url}")
                    except Exception as e:
                        st.error(f"Scraping failed: {str(e)}")
                        logger.error(f"Web scraping error: {str(e)}")
                        st.session_state.raw_text = None

        if st.session_state.raw_text:
            with st.spinner("Preprocessing text..."):
                try:
                    cleaned_text = preprocessing_agent.clean(st.session_state.raw_text)
                    st.info(f"Preprocessed text: {len(cleaned_text)} characters")
                    logger.debug(f"Preprocessed text sample: {cleaned_text[:100]}")
                except Exception as e:
                    st.error(f"Preprocessing failed: {str(e)}")
                    logger.error(f"Preprocessing error: {str(e)}")
                    cleaned_text = st.session_state.raw_text
            st.text_area("Sample of Cleaned Text (first 500 chars)", cleaned_text[:500], height=100, key="text_sample")
            with st.spinner("Chunking text..."):
                try:
                    chunks = text_chunker.chunk_text(cleaned_text)
                    st.info(f"Split text into {len(chunks)} chunks")
                    logger.debug(f"Chunk count: {len(chunks)}, first chunk sample: {chunks[0][:100] if chunks else 'None'}")
                except Exception as e:
                    st.error(f"Chunking failed: {str(e)}")
                    logger.error(f"Chunking error: {str(e)}")
                    chunks = [cleaned_text]
            if chunks:
                chunk_info = pd.DataFrame({
                    "Chunk": [f"Chunk {i+1}" for i in range(min(5, len(chunks)))],
                    "Characters": [len(chunks[i]) for i in range(min(5, len(chunks)))],
                    "Sample": [chunks[i][:100] + "..." for i in range(min(5, len(chunks)))]
                })
                st.write("Sample chunks:")
                st.dataframe(chunk_info)
                if len(chunks) > 5:
                    st.write(f"... and {len(chunks) - 5} more chunks")
            dataset_type = st.selectbox("Select Dataset Type", ("QnA", "Chain-of-Thought", "Conversational"), key="text_dataset_type")
            use_mock = st.sidebar.checkbox("Use mock generator (no API)", key="text_use_mock")
            if use_mock:
                class MockGenerator:
                    def generate(self, chunk):
                        logger.debug(f"Mock generating from chunk: {chunk[:20]}...")
                        return [{"question": f"Q{i} from {chunk[:20]}", "answer": f"A{i}"} for i in range(5)]
                    def set_target_count(self, count):
                        pass
                generator = MockGenerator()
                batch_processor = BatchProcessor(generator, max_workers=max_concurrent, max_concurrent_api_calls=10)
            else:
                if dataset_type == "QnA":
                    generator = qna_generator
                elif dataset_type == "Chain-of-Thought":
                    generator = chain_generator
                else:
                    generator = convo_generator
                batch_processor = BatchProcessor(generator, max_workers=max_concurrent, max_concurrent_api_calls=2)
            col1, col2 = st.columns(2)
            if col1.button("Generate Dataset", key="text_generate"):
                start_time = time.time()
                with st.spinner(f"Generating {dataset_type} dataset from {len(chunks)} chunks..."):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    chunk_results = []
                    processed_chunks = 0
                    combined_data = []
                    batch_size = max(1, len(chunks) // 10)
                    try:
                        for i in range(0, len(chunks), batch_size):
                            batch_chunks = chunks[i:i + batch_size]
                            target_items = max(5, target_dataset_size // max(1, len(chunks) // batch_size))
                            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch_chunks)} chunks, target_items={target_items}")
                            batch_data, batch_results = batch_processor.process_all_chunks(batch_chunks, target_items)
                            if not batch_data and not use_mock:
                                logger.warning(f"Batch {i//batch_size + 1} returned no data, using fallback")
                                batch_data = [{"question": f"Fallback Q from {chunk[:20]}", "answer": "Fallback A"} 
                                              for chunk in batch_chunks]
                            for result in batch_results:
                                if not result["success"]:
                                    st.error(f"Chunk {result['chunk_index']} failed: {result['error']}")
                                    logger.error(f"Chunk {result['chunk_index']} failed: {result['error']}")
                            combined_data.extend(batch_data)
                            chunk_results.extend(batch_results)
                            processed_chunks += len(batch_chunks)
                            progress_percent = min(1.0, processed_chunks / len(chunks))
                            progress_bar.progress(progress_percent)
                            progress_text.text(f"Processed {processed_chunks}/{len(chunks)} chunks - Generated {len(combined_data)} items")
                            time.sleep(0.1)
                        if not combined_data:
                            st.warning("No data generated, using minimal fallback dataset")
                            combined_data = [{"question": "Fallback Q", "answer": "Fallback A"}]
                            logger.warning("No data generated from chunks, applied fallback")
                    except Exception as e:
                        st.error(f"Dataset generation failed: {str(e)}")
                        logger.error(f"Generation error: {str(e)}")
                        combined_data = [{"question": "Error Q", "answer": f"Generation failed: {str(e)}"}]
                    end_time = time.time()
                    total_duration = end_time - start_time
                    items_per_second = len(combined_data) / total_duration if total_duration > 0 else 0
                    with st.spinner("Saving complete dataset..."):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        combined_file = save_csv(combined_data, dataset_type, f"combined_{timestamp}")
                        stats = {
                            "dataset_type": dataset_type,
                            "total_items": len(combined_data),
                            "total_chunks": len(chunks),
                            "processed_chunks": processed_chunks,
                            "total_duration_seconds": total_duration,
                            "items_per_second": items_per_second,
                            "timestamp": timestamp
                        }
                        save_processing_stats(stats, f"{dataset_type}_stats_{timestamp}.json")
                        st.session_state.generated_dataset_file = combined_file
                        st.session_state.dataset_type = dataset_type
                        st.session_state.combined_data = combined_data
                    st.success(f"Dataset generation complete! Generated {len(combined_data)} items in {total_duration:.2f} seconds")
                    logger.info(f"Generated {len(combined_data)} items in {total_duration:.2f} seconds")
                    st.subheader("Generation Statistics")
                    st.write(f"- Total items generated: {len(combined_data)}")
                    st.write(f"- Total processing time: {total_duration:.2f} seconds")
                    st.write(f"- Items per second: {items_per_second:.2f}")
                    st.write(f"- Chunks processed: {processed_chunks}/{len(chunks)}")
                    st.subheader("Dataset Sample")
                    st.write(combined_data[:5])
                    def get_dataset_bytes():
                        with open(st.session_state.generated_dataset_file, "rb") as f:
                            return f.read()
                    st.download_button(
                        label=f"Download {dataset_type} Dataset",
                        data=get_dataset_bytes(),
                        file_name=os.path.basename(st.session_state.generated_dataset_file),
                        key="text_dataset_download"
                    )
            if col2.button("Evaluate Dataset", disabled=st.session_state.generated_dataset_file is None, key="text_evaluate"):
                if st.session_state.generated_dataset_file and os.path.exists(st.session_state.generated_dataset_file):
                    with st.spinner(f"Evaluating {st.session_state.dataset_type} dataset..."):
                        try:
                            logger.info(f"Starting evaluation of {st.session_state.generated_dataset_file}")
                            evaluator = DatasetEvaluator(st.session_state.generated_dataset_file, st.session_state.dataset_type)
                            result = evaluator.run_evaluation()
                            if result:
                                corrected_file_name, issues, corrections, rouge_scores = result
                                st.session_state.corrected_file_name = corrected_file_name
                                with st.expander("Evaluation Results", expanded=True):
                                    if issues:
                                        st.warning(f"Issues found in the {st.session_state.dataset_type} dataset:")
                                        for issue in issues[:10]:
                                            st.write(f"- {issue}")
                                        if len(issues) > 10:
                                            st.write(f"... and {len(issues) - 10} more issues")
                                        st.success("Corrections applied:")
                                        for correction in corrections[:10]:
                                            st.write(f"- {correction}")
                                        if len(corrections) > 10:
                                            st.write(f"... and {len(corrections) - 10} more corrections")
                                    else:
                                        st.success(f"No issues found in the {st.session_state.dataset_type} dataset!")
                                    st.subheader("ROUGE Scores")
                                    st.write(f"- ROUGE-1: {rouge_scores['rouge1']:.3f}")
                                    st.write(f"- ROUGE-2: {rouge_scores['rouge2']:.3f}")
                                    st.write(f"- ROUGE-L: {rouge_scores['rougeL']:.3f}")
                                def get_corrected_dataset_bytes():
                                    with open(st.session_state.corrected_file_name, "rb") as f:
                                        return f.read()
                                if os.path.exists(corrected_file_name):
                                    st.download_button(
                                        label="Download Corrected Dataset",
                                        data=get_corrected_dataset_bytes(),
                                        file_name=os.path.basename(corrected_file_name),
                                        key="text_corrected_dataset_download"
                                    )
                                else:
                                    st.error(f"Corrected file not found at: {corrected_file_name}")
                            else:
                                st.error("Failed to evaluate the dataset. Evaluator returned None.")
                                logger.error("Evaluator.run_evaluation() returned None")
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                            logger.error(f"Evaluation error: {str(e)}", exc_info=True)
                else:
                    st.error("No dataset file found. Please generate a dataset first.")
                    logger.error(f"Evaluation attempted but no file found at {st.session_state.generated_dataset_file}")
        else:
            st.info("Please extract text from a PDF or scrape a website to proceed.")

# Image Dataset Tab
with tab2:
    st.subheader("Image Dataset Generator with Augmentation")
    prompt = st.text_input("Enter image category (e.g., 'cat', 'dog'):", key="image_prompt")
    num_images = st.number_input("How many unique images do you need?", min_value=10, max_value=10000, value=1000, step=10, key="image_num")
    if st.button("Generate Image Dataset", key="image_generate"):
        if prompt and num_images:
            with st.spinner(f"Fetching {num_images} unique images and applying augmentation..."):
                total_images, dataset_folder = generate_image_dataset(prompt, num_images)
                st.success(f"Dataset generated with {total_images} images in '{dataset_folder}'!")
                logger.info(f"Generated image dataset with {total_images} images in {dataset_folder}")
                st.write(f"Images saved to: {dataset_folder}")
                # Note: Streamlit doesn't natively support downloading folders; users need to access the folder manually or zip it
                st.info("To download the dataset, please zip the folder manually from your file system.")

if __name__ == "__main__":
    logger.info("Streamlit app started")