import requests
from bs4 import BeautifulSoup
import logging
import time
from urllib.parse import urljoin, urlparse

class WebsiteScrapingAgent:
    """Agent for scraping text content from websites with improved capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.visited_urls = set()
        self.max_depth = 1
        self.max_pages = 5
        self.max_chars = 50000  # Maximum character limit
        self.total_chars = 0    # Track total characters scraped
    
    def set_max_depth(self, depth):
        """Set the maximum depth for internal links to follow."""
        self.max_depth = depth
    
    def set_max_pages(self, pages):
        """Set the maximum number of pages to scrape."""
        self.max_pages = pages
    
    def set_max_chars(self, chars):
        """Set the maximum number of characters to scrape."""
        self.max_chars = chars
    
    def is_same_domain(self, url, base_url):
        """Check if URL belongs to the same domain as the base URL."""
        return urlparse(url).netloc == urlparse(base_url).netloc
    
    def extract_content(self, html_content):
        """Extract readable text content from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, header, footer, and nav elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        self.logger.debug(f"Extracted {len(text)} characters from HTML content")
        return text
    
    def get_internal_links(self, html_content, base_url):
        """Extract internal links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(base_url, href)
            if self.is_same_domain(absolute_url, base_url) and '#' not in absolute_url:
                links.append(absolute_url)
        
        unique_links = list(set(links))
        self.logger.debug(f"Found {len(unique_links)} internal links")
        return unique_links
    
    def scrape_url(self, url):
        """Scrape content from a single URL."""
        try:
            self.logger.info(f"Scraping {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            text_content = self.extract_content(html_content)
            
            self.logger.info(f"Successfully scraped {len(text_content)} characters from {url}")
            return text_content, html_content
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return "", ""
    
    def scrape_recursive(self, start_url, current_depth=0):
        """Recursively scrape a website up to max_depth or max_chars."""
        if (current_depth > self.max_depth or 
            len(self.visited_urls) >= self.max_pages or 
            self.total_chars >= self.max_chars):
            self.logger.debug(f"Stopping recursion: depth={current_depth}, pages={len(self.visited_urls)}, chars={self.total_chars}")
            return ""
        
        if start_url in self.visited_urls:
            self.logger.debug(f"Skipping already visited URL: {start_url}")
            return ""
        
        self.visited_urls.add(start_url)
        text_content, html_content = self.scrape_url(start_url)
        
        # Trim content if it exceeds remaining character limit
        remaining_chars = self.max_chars - self.total_chars
        self.logger.debug(f"Before trim: {len(text_content)} chars from {start_url}, remaining={remaining_chars}")
        if len(text_content) > remaining_chars:
            text_content = text_content[:remaining_chars]
            self.logger.debug(f"Trimmed to {len(text_content)} chars")
        
        self.total_chars += len(text_content)
        all_text = text_content
        self.logger.debug(f"After adding: total_chars={self.total_chars}")
        
        # Continue recursion if limits allow
        if (current_depth < self.max_depth and 
            len(self.visited_urls) < self.max_pages and 
            self.total_chars < self.max_chars):
            internal_links = self.get_internal_links(html_content, start_url)
            
            for link in internal_links:
                if (link not in self.visited_urls and 
                    len(self.visited_urls) < self.max_pages and 
                    self.total_chars < self.max_chars):
                    time.sleep(1)  # Respectful delay
                    additional_text = self.scrape_recursive(link, current_depth + 1)
                    remaining_chars = self.max_chars - self.total_chars
                    if additional_text and remaining_chars > 0:
                        additional_text = additional_text[:remaining_chars]
                        all_text += "\n\n" + additional_text
                        self.total_chars += len(additional_text)
                        self.logger.debug(f"Added {len(additional_text)} chars from {link}, total now={self.total_chars}")
        
        return all_text
    
    def scrape(self, url, max_depth=1, max_pages=5, max_chars=50000):
        """Main method to scrape content from a website."""
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_chars = max_chars
        self.visited_urls = set()
        self.total_chars = 0
        
        # Handle Wikipedia URLs
        if "wikipedia.org" in url:
            modified_url = self._handle_wikipedia(url)
            if modified_url:
                url = modified_url
        
        self.logger.info(f"Starting scrape of {url} with max_depth={max_depth}, max_pages={max_pages}, max_chars={max_chars}")
        content = self.scrape_recursive(url)
        
        if not content:
            self.logger.warning(f"No content scraped from {url}")
            raise ValueError(f"Failed to scrape any content from {url}")
        
        self.logger.info(f"Completed scraping with {len(self.visited_urls)} pages and {len(content)} characters")
        self.logger.debug(f"Final content sample: {content[:100] if content else 'None'}")
        return content
    
    def _handle_wikipedia(self, url):
        """Special handling for Wikipedia URLs to get printer-friendly version."""
        try:
            if "wikipedia.org/wiki/" in url:
                parts = url.split("/wiki/")
                if len(parts) > 1:
                    return parts[0] + "/w/index.php?title=" + parts[1] + "&printable=yes"
        except Exception as e:
            self.logger.warning(f"Error processing Wikipedia URL: {str(e)}")
        return None

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    agent = WebsiteScrapingAgent()
    try:
        content = agent.scrape("https://en.wikipedia.org/wiki/Stuart_J._Russell")
        print(f"Scraped {len(content)} characters")
        print(f"Content sample: {content[:100] if content else 'None'}")
        # Optionally write to a file to verify content
        with open("scraped_content.txt", "w", encoding="utf-8") as f:
            f.write(content)
    except ValueError as e:
        print(f"Error: {e}")