# batch_processor.py
import concurrent.futures
import logging
import time
import random
from threading import Semaphore
from tqdm import tqdm

class BatchProcessor:
    def __init__(self, generator, max_workers=3, retry_attempts=3, retry_delay=2, max_concurrent_api_calls=2):
        self.generator = generator
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        # Semaphore to limit concurrent API calls (adjust based on Groq's rate limit)
        self.api_semaphore = Semaphore(max_concurrent_api_calls)

    def process_chunk(self, chunk, target_items):
        if hasattr(self.generator, 'set_target_count'):
            self.generator.set_target_count(target_items)
        
        with self.api_semaphore:  # Limit concurrent API calls
            for attempt in range(self.retry_attempts):
                try:
                    start_time = time.time()
                    result = self.generator.generate(chunk)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    items_count = len(result) if result else 0
                    items_per_second = items_count / processing_time if processing_time > 0 else 0
                    
                    self.logger.info(f"Chunk processed successfully: {items_count} items in {processing_time:.2f}s ({items_per_second:.2f} items/s)")
                    
                    return result, {
                        "success": True, 
                        "chunk_length": len(chunk), 
                        "items_generated": items_count,
                        "processing_time": processing_time,
                        "items_per_second": items_per_second
                    }
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt+1}/{self.retry_attempts} failed: {str(e)}")
                    if attempt < self.retry_attempts - 1:
                        jitter = random.uniform(0.5, 1.5)
                        sleep_time = self.retry_delay * (attempt + 1) * jitter
                        self.logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        self.logger.error(f"All {self.retry_attempts} attempts failed for chunk of length {len(chunk)}: {str(e)}")
                        return [], {"success": False, "error": str(e), "chunk_length": len(chunk)}

    def process_all_chunks(self, chunks, target_items):
        all_results = []
        chunk_stats = []
        
        self.logger.info(f"Processing {len(chunks)} chunks with max_workers={self.max_workers}, max_concurrent_api_calls={self.api_semaphore._value}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, target_items): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result, stats = future.result()
                    all_results.extend(result)
                    chunk_stats.append(stats)
                    chunk_stats[-1]["chunk_index"] = chunk_index
                except Exception as e:
                    self.logger.error(f"Unexpected error processing chunk {chunk_index}: {str(e)}")
                    chunk_stats.append({
                        "chunk_index": chunk_index,
                        "success": False,
                        "error": str(e),
                        "chunk_length": len(chunks[chunk_index]) if chunks and chunk_index < len(chunks) else 0
                    })
        
        successful_chunks = sum(1 for stat in chunk_stats if stat.get("success", False))
        total_items = sum(stat.get("items_generated", 0) for stat in chunk_stats)
        
        self.logger.info(f"Processed {successful_chunks}/{len(chunks)} chunks successfully")
        self.logger.info(f"Generated {total_items} items total")
        
        return all_results, chunk_stats