"""
Nhatot.com Real Estate Crawler
Optimized for nhatot.com structure with hashed classes
"""

import time
import random
import pandas as pd
from loguru import logger
import json
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Loguru is auto-configured with beautiful colored output!

class NhatotScraper:
    def __init__(self, output_file: str = 'nhatot_data.csv'):
        self.output_file = output_file
        self.data_buffer: List[Dict] = []
        self.driver = self._setup_driver()

    def _setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome with stealth settings"""
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Enabled for speed
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        chrome_options.add_argument(f'user-agent={user_agent}')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Remove webdriver flag
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
        
        logger.info("Driver initialized successfully")
        return driver

    def _random_sleep(self, min_s: float = 2.0, max_s: float = 5.0):
        """Random sleep to mimic human behavior"""
        time.sleep(random.uniform(min_s, max_s))

    def get_listing_urls(self, search_url: str, max_pages: int = 1) -> List[str]:
        """Extract listing URLs from search pages using JSON-LD"""
        all_urls = set()
        consecutive_empty_pages = 0  # Track consecutive empty pages
        
        for page_num in range(1, max_pages + 1):
            if page_num == 1:
                target_url = search_url
            else:
                target_url = f"{search_url}?page={page_num}"
            
            logger.info(f"Fetching page {page_num}: {target_url}")
            
            try:
                self.driver.get(target_url)
                self._random_sleep(5, 8)  # Wait for JS to load (balanced timing)
                
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Method 1: Extract from JSON-LD schema
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
                            logger.info(f"Found {len(page_urls)} URLs from JSON-LD")
                            break
                    except:
                        continue
                
                # Method 2: Fallback - find links in HTML
                if not page_urls:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if 'mua-ban-nha-dat' in href and '.htm' in href:
                            full_url = href if href.startswith('http') else f"https://www.nhatot.com{href}"
                            if full_url not in all_urls:
                                all_urls.add(full_url)
                                page_urls.append(full_url)
                    if page_urls:
                        logger.info(f"Found {len(page_urls)} URLs from HTML")
                
                # Check if page is empty
                if not page_urls:
                    consecutive_empty_pages += 1
                    logger.warning(f"No URLs found on page {page_num} ({consecutive_empty_pages}/3 consecutive empty)")
                    
                    # Only break after 3 consecutive empty pages
                    if consecutive_empty_pages >= 3:
                        logger.info("Stopping: 3 consecutive empty pages")
                        break
                else:
                    consecutive_empty_pages = 0  # Reset counter when we find URLs
                    
            except Exception as e:
                logger.error(f"Error fetching page {page_num}: {e}")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                    break
        
        return list(all_urls)

    def parse_detail_page(self, url: str, retry_count: int = 0) -> Optional[Dict]:
        """
        Parse detail page using itemprop attributes (semantic HTML)
        Much more reliable than hashed CSS classes!
        """
        logger.info(f"Parsing: {url}")
        
        # Retry logic if browser crashed
        if retry_count > 2:
            logger.error(f"Max retries reached for {url}")
            return {
                'Giá bán': '', 'Thành phố': '', 'Phường/Xã': '', 'Diện tích (m2)': '',
                'Loại hình': '', 'Giấy tờ pháp lý': '', 'Hướng': '',
                'Chiều ngang (m)': '', 'Chiều dài (m)': '', 'Số phòng ngủ': '',
                'Số phòng vệ sinh': '', 'Số tầng': '', 'Tình trạng nội thất': ''
            }
        
        try:
            self.driver.get(url)
            self._random_sleep(2, 3)  # Reduced for speed
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Initialize all fields with empty string (as requested by user)
            item = {
                'Giá bán': '',
                'Thành phố': '',
                'Phường/Xã': '',
                'Diện tích (m2)': '',
                'Loại hình': '',
                'Giấy tờ pháp lý': '',
                'Hướng': '',
                'Chiều ngang (m)': '',
                'Chiều dài (m)': '',
                'Số phòng ngủ': '',
                'Số phòng vệ sinh': '',
                'Số tầng': '',
                'Tình trạng nội thất': ''
            }
            
            # 1. Giá bán - Using class selector
            price_elem = soup.find('b', class_='pyhk1dv')
            if price_elem:
                item['Giá bán'] = price_elem.text.strip()
            
            # 2. Loại hình - Using itemprop
            house_type_elem = soup.find('strong', itemprop='house_type')
            if house_type_elem:
                item['Loại hình'] = house_type_elem.text.strip()
            
            # 3. Địa chỉ - Extract from span with class
            # Format: "Đường X, Phường Y, Quận Z, Tp ABC"
            address_elem = soup.find('span', class_='tunpaa5')
            if address_elem:
                full_address = address_elem.text.strip()
                # Parse address: usually "Đường, Phường, Quận, Thành phố"
                parts = [p.strip() for p in full_address.split(',')]
                if len(parts) >= 4:
                    item['Phường/Xã'] = parts[-3]  # Phường/Quận
                    item['Thành phố'] = parts[-1]  # Thành phố
                elif len(parts) >= 1:
                    item['Thành phố'] = parts[-1]
            
            # 4. Diện tích - Using itemprop
            size_elem = soup.find('strong', itemprop='size')
            if size_elem:
                item['Diện tích (m2)'] = size_elem.text.strip().replace('m²', '').replace('m2', '').strip()
            
            # 5. Giấy tờ pháp lý - Using itemprop
            legal_elem = soup.find('strong', itemprop='property_legal_document')
            if legal_elem:
                item['Giấy tờ pháp lý'] = legal_elem.text.strip()
            
            # 6. Số phòng ngủ - Using itemprop
            rooms_elem = soup.find('strong', itemprop='rooms')
            if rooms_elem:
                item['Số phòng ngủ'] = rooms_elem.text.strip().replace('phòng', '').strip()
            
            # 7. Số phòng vệ sinh - Using itemprop
            toilets_elem = soup.find('strong', itemprop='toilets')
            if toilets_elem:
                item['Số phòng vệ sinh'] = toilets_elem.text.strip().replace('phòng', '').strip()
            
            # 8. Chiều ngang - Using itemprop
            width_elem = soup.find('strong', itemprop='width')
            if width_elem:
                item['Chiều ngang (m)'] = width_elem.text.strip().replace('m', '').strip()
            
            # 9. Chiều dài - Using itemprop
            length_elem = soup.find('strong', itemprop='length')
            if length_elem:
                item['Chiều dài (m)'] = length_elem.text.strip().replace('m', '').strip()
            
            # 10. Số tầng - Using itemprop
            floors_elem = soup.find('strong', itemprop='floors')
            if floors_elem:
                item['Số tầng'] = floors_elem.text.strip()
            
            # 11. Hướng - Using itemprop
            direction_elem = soup.find('strong', itemprop='direction')
            if direction_elem:
                item['Hướng'] = direction_elem.text.strip()
            
            # 12. Tình trạng nội thất - Using itemprop
            furnishing_elem = soup.find('strong', itemprop='furnishing_sell')
            if furnishing_elem:
                item['Tình trạng nội thất'] = furnishing_elem.text.strip()
            
            # Count filled fields (non-empty)
            filled = sum(1 for v in item.values() if v)
            logger.info(f"✓ Extracted {filled}/{len(item)} fields | Price: {item['Giá bán'] or 'N/A'}")
            
            return item
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error parsing {url}: {error_msg}")
            
            # If browser crashed, restart it and retry
            if 'invalid session id' in error_msg or 'disconnected' in error_msg:
                logger.warning(f"Browser crashed! Restarting... (retry {retry_count + 1}/3)")
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = self._setup_driver()
                self._random_sleep(2, 4)
                return self.parse_detail_page(url, retry_count + 1)
            
            # Return empty dict with all fields
            return {
                'Giá bán': '', 'Thành phố': '', 'Phường/Xã': '', 'Diện tích (m2)': '',
                'Loại hình': '', 'Giấy tờ pháp lý': '', 'Hướng': '',
                'Chiều ngang (m)': '', 'Chiều dài (m)': '', 'Số phòng ngủ': '',
                'Số phòng vệ sinh': '', 'Số tầng': '', 'Tình trạng nội thất': ''
            }

    def run(self, start_url: str, pages: int = 100):
        """Main crawling workflow"""
        try:
            # Step 1: Get listing URLs
            logger.info(f"Starting crawl from: {start_url}")
            logger.info(f"Will attempt to crawl up to {pages} pages")
            listing_urls = self.get_listing_urls(start_url, pages)
            logger.info(f"Total URLs collected: {len(listing_urls)}")
            
            if not listing_urls:
                logger.error("No URLs found! Check if site is blocking.")
                return
            
            # Step 2: Parse each detail page
            for idx, url in enumerate(listing_urls):
                logger.info(f"[{idx+1}/{len(listing_urls)}] Processing...")
                try:
                    data = self.parse_detail_page(url)
                    
                    if data:
                        self.data_buffer.append(data)
                    
                    # Save every 5 items (more frequent)
                    if len(self.data_buffer) % 5 == 0 and self.data_buffer:
                        self._save_to_csv()
                    
                    self._random_sleep(2, 4)  # Reduced for speed
                except KeyboardInterrupt:
                    logger.warning("User interrupted! Saving data...")
                    self._save_to_csv()
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error on {url}: {e}")
                    continue
            
            # Final save
            self._save_to_csv()
            logger.info(f"✅ Crawling completed! Total records: {len(self.data_buffer)}")
            
        except Exception as e:
            logger.critical(f"Critical error: {e}")
        finally:
            self.driver.quit()

    def _save_to_csv(self):
        """Save data to CSV"""
        if not self.data_buffer:
            return
        
        df = pd.DataFrame(self.data_buffer)
        
        # Reorder columns (no need to drop url/title, they don't exist anymore)
        column_order = [
            'Giá bán', 
            'Thành phố', 
            'Phường/Xã', 
            'Diện tích (m2)', 
            'Loại hình', 
            'Giấy tờ pháp lý', 
            'Hướng', 
            'Chiều ngang (m)',
            'Chiều dài (m)',
            'Số phòng ngủ', 
            'Số phòng vệ sinh', 
            'Số tầng',
            'Tình trạng nội thất'
        ]
        
        existing_cols = [c for c in column_order if c in df.columns]
        df = df[existing_cols]
        
        df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Saved {len(df)} records to {self.output_file}")

if __name__ == "__main__":
    scraper = NhatotScraper(output_file='nhatot_real_estate.csv')
    # Set pages=100 to crawl all available pages (will stop when no more pages found)
    scraper.run("https://www.nhatot.com/mua-ban-nha-dat", pages=100)
