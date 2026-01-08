"""
Test script for Crawl4AI Nhatot scraper
Quick test to verify the implementation works correctly
"""

import asyncio
from crawl_nhatot_crawl4ai import Crawl4AINhatotScraper
from loguru import logger

async def test_scraper():
    """Test the scraper with 1 page"""
    logger.info("🧪 Testing Crawl4AI scraper with 1 page...")
    
    scraper = Crawl4AINhatotScraper(
        output_file='nhatot_test.csv',
        max_concurrent=3
    )
    
    # Test with just 1 page
    await scraper.run("https://www.nhatot.com/mua-ban-nha-dat", pages=1)
    
    logger.success("✅ Test completed! Check nhatot_test.csv for results")

if __name__ == "__main__":
    asyncio.run(test_scraper())
