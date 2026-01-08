"""
Nhatot.com Real Estate Crawler - Powered by Crawl4AI
Modern async crawler with browser pooling and intelligent extraction
"""

import asyncio
import pandas as pd
from loguru import logger
import json
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import time

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


class Crawl4AINhatotScraper:
    def __init__(self, output_file: str = 'nhatot_crawl4ai.csv', max_concurrent: int = 5):
        """
        Modern async scraper using Crawl4AI
        
        Args:
            output_file: Output CSV file path
            max_concurrent: Number of concurrent browser pages (default: 5)
        """
        self.output_file = output_file
        self.max_concurrent = max_concurrent
        self.data_buffer: List[Dict] = []
        
        logger.info(f"🚀 Initializing Crawl4AI scraper with {max_concurrent} concurrent pages")

    async def get_listing_urls(self, crawler, search_url: str, max_pages: int = 1) -> List[str]:
        """
        Extract listing URLs from search pages using JSON-LD
        
        Args:
            crawler: AsyncWebCrawler instance
            search_url: Base search URL
            max_pages: Maximum number of pages to crawl
        """
        all_urls = set()
        consecutive_empty_pages = 0
        
        logger.info(f"📋 Collecting URLs from up to {max_pages} pages...")
        
        for page_num in range(1, max_pages + 1):
            if page_num == 1:
                target_url = search_url
            else:
                target_url = f"{search_url}?page={page_num}"
            
            logger.info(f"📄 Fetching page {page_num}/{max_pages}: {target_url}")
            
            try:
                # Crawl the search page
                result = await crawler.arun(
                    url=target_url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,  # Always fetch fresh
                        page_timeout=60000,  # 60 seconds
                        wait_for="networkidle",  # Wait for network to be idle
                        delay_before_return_html=3.0,  # Additional wait after network idle
                        js_code="window.scrollTo(0, document.body.scrollHeight);",  # Scroll to trigger lazy load
                    )
                )
                
                if not result.success:
                    logger.warning(f"Failed to fetch page {page_num}: {result.error_message}")
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= 3:
                        break
                    continue
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Extract URLs from JSON-LD
                json_scripts = soup.find_all('script', type='application/ld+json')
                page_urls = []
                
                for script in json_scripts:
                    try:
                        data = json.loads(script.string)
                        if isinstance(data, dict) and data.get('@type') == 'ItemList':
                            items = data.get('itemListElement', [])
                            for item in items:
                                url = item.get('url')
                                if url and '.htm' in url and url not in all_urls:
                                    all_urls.add(url)
                                    page_urls.append(url)
                            logger.info(f"✓ Found {len(page_urls)} URLs on page {page_num}")
                            break
                    except Exception as e:
                        logger.debug(f"Error parsing JSON-LD: {e}")
                        continue
                
                # Check if page is empty
                if not page_urls:
                    consecutive_empty_pages += 1
                    logger.warning(f"No URLs found on page {page_num} ({consecutive_empty_pages}/3)")
                    if consecutive_empty_pages >= 3:
                        logger.info("Stopping: 3 consecutive empty pages")
                        break
                else:
                    consecutive_empty_pages = 0
                    
            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                    break
        
        logger.info(f"📊 Total URLs collected: {len(all_urls)}")
        return list(all_urls)

    async def parse_detail_page(self, crawler, url: str) -> Optional[Dict]:
        """
        Parse detail page to extract property information
        
        Args:
            crawler: AsyncWebCrawler instance
            url: Property detail URL
        """
        try:
            # Crawl the detail page
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    page_timeout=60000,  # Increase timeout
                    wait_for="networkidle",  # Wait for network to be idle
                    delay_before_return_html=2.0,  # Additional wait
                    js_code="window.scrollTo(0, 500);",  # Scroll to trigger content load
                )
            )
            
            if not result.success:
                logger.warning(f"Failed to fetch {url}: {result.error_message}")
                return self._empty_item()
            
            # Parse HTML
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Initialize item
            item = {
                'Giá bán': '', 'Thành phố': '', 'Phường/Xã': '', 'Diện tích (m2)': '',
                'Loại hình': '', 'Giấy tờ pháp lý': '', 'Hướng': '',
                'Chiều ngang (m)': '', 'Chiều dài (m)': '', 'Số phòng ngủ': '',
                'Số phòng vệ sinh': '', 'Số tầng': '', 'Tình trạng nội thất': ''
            }
            
            # Extract data using selectors
            # 1. Giá bán
            price_elem = soup.find('b', class_='pyhk1dv')
            if price_elem:
                item['Giá bán'] = price_elem.text.strip()
            
            # 2. Loại hình
            house_type_elem = soup.find('strong', itemprop='house_type')
            if house_type_elem:
                item['Loại hình'] = house_type_elem.text.strip()
            
            # 3. Địa chỉ
            address_elem = soup.find('span', class_='tunpaa5')
            if address_elem:
                full_address = address_elem.text.strip()
                parts = [p.strip() for p in full_address.split(',')]
                if len(parts) >= 4:
                    item['Phường/Xã'] = parts[-3]
                    item['Thành phố'] = parts[-1]
                elif len(parts) >= 1:
                    item['Thành phố'] = parts[-1]
            
            # 4. Diện tích
            size_elem = soup.find('strong', itemprop='size')
            if size_elem:
                item['Diện tích (m2)'] = size_elem.text.strip().replace('m²', '').replace('m2', '').strip()
            
            # 5. Giấy tờ pháp lý
            legal_elem = soup.find('strong', itemprop='property_legal_document')
            if legal_elem:
                item['Giấy tờ pháp lý'] = legal_elem.text.strip()
            
            # 6. Số phòng ngủ
            rooms_elem = soup.find('strong', itemprop='rooms')
            if rooms_elem:
                item['Số phòng ngủ'] = rooms_elem.text.strip().replace('phòng', '').strip()
            
            # 7. Số phòng vệ sinh
            toilets_elem = soup.find('strong', itemprop='toilets')
            if toilets_elem:
                item['Số phòng vệ sinh'] = toilets_elem.text.strip().replace('phòng', '').strip()
            
            # 8. Chiều ngang
            width_elem = soup.find('strong', itemprop='width')
            if width_elem:
                item['Chiều ngang (m)'] = width_elem.text.strip().replace('m', '').strip()
            
            # 9. Chiều dài
            length_elem = soup.find('strong', itemprop='length')
            if length_elem:
                item['Chiều dài (m)'] = length_elem.text.strip().replace('m', '').strip()
            
            # 10. Số tầng
            floors_elem = soup.find('strong', itemprop='floors')
            if floors_elem:
                item['Số tầng'] = floors_elem.text.strip()
            
            # 11. Hướng
            direction_elem = soup.find('strong', itemprop='direction')
            if direction_elem:
                item['Hướng'] = direction_elem.text.strip()
            
            # 12. Tình trạng nội thất
            furnishing_elem = soup.find('strong', itemprop='furnishing_sell')
            if furnishing_elem:
                item['Tình trạng nội thất'] = furnishing_elem.text.strip()
            
            # Log extraction success
            filled = sum(1 for v in item.values() if v)
            logger.debug(f"✓ {filled}/13 fields | {item.get('Giá bán', 'N/A')}")
            
            return item
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {str(e)[:100]}")
            return self._empty_item()

    def _empty_item(self) -> Dict:
        """Return empty item with all fields"""
        return {
            'Giá bán': '', 'Thành phố': '', 'Phường/Xã': '', 'Diện tích (m2)': '',
            'Loại hình': '', 'Giấy tờ pháp lý': '', 'Hướng': '',
            'Chiều ngang (m)': '', 'Chiều dài (m)': '', 'Số phòng ngủ': '',
            'Số phòng vệ sinh': '', 'Số tầng': '', 'Tình trạng nội thất': ''
        }

    def _save_to_csv(self):
        """Save data buffer to CSV"""
        if not self.data_buffer:
            return
        
        df = pd.DataFrame(self.data_buffer)
        
        column_order = [
            'Giá bán', 'Thành phố', 'Phường/Xã', 'Diện tích (m2)', 
            'Loại hình', 'Giấy tờ pháp lý', 'Hướng', 
            'Chiều ngang (m)', 'Chiều dài (m)', 'Số phòng ngủ', 
            'Số phòng vệ sinh', 'Số tầng', 'Tình trạng nội thất'
        ]
        
        existing_cols = [c for c in column_order if c in df.columns]
        df = df[existing_cols]
        
        df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Saved {len(df)} records to {self.output_file}")

    async def run(self, start_url: str, pages: int = 100):
        """
        Main async crawling workflow
        
        Args:
            start_url: Search URL to start from
            pages: Number of pages to crawl
        """
        start_time = time.time()
        
        # Configure browser with stealth mode to bypass anti-bot
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport_width=1920,
            viewport_height=1080,
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            use_managed_browser=True,  # Use managed browser for better stealth
            use_persistent_context=False,
        )
        
        # Create crawler with browser pooling
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                # Step 1: Collect listing URLs
                logger.info(f"🎯 Starting crawl from: {start_url}")
                listing_urls = await self.get_listing_urls(crawler, start_url, pages)
                
                if not listing_urls:
                    logger.error("❌ No URLs found!")
                    return
                
                # Step 2: Parse detail pages with concurrency control
                logger.info(f"🚀 Processing {len(listing_urls)} listings with {self.max_concurrent} concurrent pages...")
                
                # Process in batches to control concurrency
                for i in range(0, len(listing_urls), self.max_concurrent):
                    batch = listing_urls[i:i + self.max_concurrent]
                    batch_num = i // self.max_concurrent + 1
                    total_batches = (len(listing_urls) + self.max_concurrent - 1) // self.max_concurrent
                    
                    logger.info(f"📦 Batch {batch_num}/{total_batches} ({len(batch)} URLs)")
                    
                    # Process batch concurrently
                    tasks = [self.parse_detail_page(crawler, url) for url in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Collect results
                    for result in results:
                        if isinstance(result, dict):
                            self.data_buffer.append(result)
                        elif isinstance(result, Exception):
                            logger.error(f"Task failed: {result}")
                    
                    # Save periodically
                    if (i + self.max_concurrent) % 20 == 0 or (i + self.max_concurrent) >= len(listing_urls):
                        self._save_to_csv()
                        progress = min(i + self.max_concurrent, len(listing_urls))
                        logger.info(f"Progress: {progress}/{len(listing_urls)} ({progress*100//len(listing_urls)}%)")
                
                # Final save
                self._save_to_csv()
                
                elapsed = time.time() - start_time
                logger.success(f"✅ Completed! {len(self.data_buffer)} records in {elapsed/60:.1f} minutes")
                logger.success(f"⚡ Average: {elapsed/len(listing_urls):.2f}s per listing")
                
            except Exception as e:
                logger.critical(f"Critical error: {e}")
                self._save_to_csv()  # Save what we have


async def main():
    """Entry point for async execution"""
    # Configuration
    MAX_CONCURRENT = 10  # Number of concurrent pages (adjust based on your system)
    MAX_PAGES = 310  # Number of search pages to crawl
    
    scraper = Crawl4AINhatotScraper(
        output_file='nhatot_crawl4ai.csv',
        max_concurrent=MAX_CONCURRENT
    )
    
    await scraper.run("https://www.nhatot.com/mua-ban-nha-dat", pages=MAX_PAGES)


if __name__ == "__main__":
    # Run the async scraper
    asyncio.run(main())
