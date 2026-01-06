# 🏠 PredictHousing - Real Estate Data Crawler

A professional web scraping tool for collecting real estate data from **Nhatot.com** (Vietnam's leading classifieds platform) using Selenium and BeautifulSoup.

## 📋 Features

- ✅ **Automated data collection** from nhatot.com real estate listings
- ✅ **Semantic HTML parsing** using `itemprop` attributes for reliable extraction
- ✅ **Retry mechanism** with automatic browser restart on failures
- ✅ **Headless mode** for faster performance
- ✅ **Auto-save** functionality to prevent data loss
- ✅ **Comprehensive field extraction**: Price, Location, Area, Property Type, Legal Documents, Direction, Dimensions, Rooms, Floors, Furnishing
- ✅ **Beautiful logging** with Loguru
- ✅ **Pagination support** with smart empty-page detection

## 🚀 Quick Start

### Prerequisites

```bash
pip install loguru pandas selenium beautifulsoup4 webdriver-manager
```

### Basic Usage

```bash
python crawl_nhatot.py
```

The script will:
1. Crawl up to 100 pages from nhatot.com
2. Extract all listing URLs
3. Parse detailed information from each listing
4. Save data to `nhatot_real_estate.csv`
5. Auto-save every 5 records

## 📊 Data Fields

The following fields are extracted from each listing:

| Field | Description | Example |
|-------|-------------|---------|
| **Giá bán** | Sale price | "15,5 tỷ" |
| **Thành phố** | City/Province | "Tp Hồ Chí Minh" |
| **Phường/Xã** | District/Ward | "Quận 1" |
| **Diện tích (m2)** | Land area | "68" |
| **Loại hình** | Property type | "Nhà mặt phố, mặt tiền" |
| **Giấy tờ pháp lý** | Legal documents | "Đã có sổ" |
| **Hướng** | Direction | "Đông Nam" |
| **Chiều ngang (m)** | Width | "7" |
| **Chiều dài (m)** | Length | "10" |
| **Số phòng ngủ** | Bedrooms | "3" |
| **Số phòng vệ sinh** | Bathrooms | "2" |
| **Số tầng** | Floors | "2" |
| **Tình trạng nội thất** | Furnishing status | "Hoàn thiện cơ bản" |

## 🛠️ Configuration

### Adjust Crawling Speed

Edit `crawl_nhatot.py`:

```python
# Line 72: Listing pages wait time
self._random_sleep(4, 6)  # Increase for more reliability, decrease for speed

# Line 148: Detail pages wait time  
self._random_sleep(2, 3)  # Adjust based on your connection speed
```

### Headless Mode

```python
# Line 34: Toggle headless mode
chrome_options.add_argument("--headless=new")  # Comment out to see browser
```

### Max Pages

```python
# Line 343: Set maximum pages to crawl
scraper.run("https://www.nhatot.com/mua-ban-nha-dat", pages=100)
```

## 🔧 Technical Details

### Architecture

- **Selenium WebDriver**: Handles JavaScript-rendered pages
- **BeautifulSoup**: Parses HTML and extracts data
- **Loguru**: Beautiful logging with colors and timestamps
- **Pandas**: Data manipulation and CSV export

### Anti-Detection Features

1. **Stealth settings**: Removes automation flags
2. **Random sleep intervals**: Mimics human behavior
3. **User-Agent rotation**: Appears as legitimate browser
4. **CDP commands**: Hides `navigator.webdriver` property

### Error Handling

- **Browser crash recovery**: Auto-restarts driver and retries (max 3 attempts)
- **Keyboard interrupt**: Saves data before exiting (Ctrl+C)
- **Empty page detection**: Stops after 3 consecutive empty pages
- **Per-record error handling**: Continues crawling even if individual pages fail

## 📈 Performance

### Speed Optimization

| Configuration | Speed | Reliability |
|---------------|-------|-------------|
| Default (headless + optimized timing) | ~2.5s per listing | ⭐⭐⭐⭐⭐ |
| Non-headless | ~5s per listing | ⭐⭐⭐⭐⭐ |
| Very fast (2s sleep) | ~1.5s per listing | ⭐⭐⭐ (may miss data) |

**Estimated time for 100 listings**: ~5-7 minutes

## 📁 Output Format

CSV file with UTF-8-BOM encoding (Excel-compatible):

```csv
Giá bán,Thành phố,Phường/Xã,Diện tích (m2),Loại hình,...
"15,5 tỷ",Tp Hồ Chí Minh,Quận 1,68,"Nhà mặt phố, mặt tiền",...
"1,7 tỷ",Bình Dương,Phường Bình Hòa,60,"Nhà ngõ, hẻm",...
```

## 🐛 Troubleshooting

### Issue: "No URLs found on page 2, 3, 4"

**Solution**: JavaScript didn't load in time. Increase wait time:
```python
self._random_sleep(5, 8)  # Line 72
```

### Issue: "Browser crashed" errors

**Solution**: Automatically handled by retry mechanism. If persists:
1. Update Chrome browser
2. Update chromedriver: `pip install --upgrade webdriver-manager`

### Issue: "Extracted 0/13 fields"

**Cause**: Page structure changed or Cloudflare blocked request
**Solution**: 
1. Check if website is accessible manually
2. May need to update selectors if site redesigned

## 🔒 Legal & Ethical Considerations

- ⚠️ **Respect robots.txt**: Check website's crawling policy
- ⚠️ **Rate limiting**: Built-in delays to avoid server overload
- ⚠️ **Data usage**: For educational/research purposes only
- ⚠️ **Terms of Service**: Ensure compliance with nhatot.com's ToS

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Multi-threading support
- [ ] Database storage (PostgreSQL/MongoDB)
- [ ] More detailed error reporting Dashboard
- [ ] Export to multiple formats (JSON, Excel, SQLite)

## 📝 License

This project is for educational purposes. Please ensure compliance with local laws and website terms of service before use.

## 👨‍💻 Author

**DUT-AI Team**  
University Project - Data Science & Machine Learning

---

### 📚 Dependencies

```txt
loguru>=0.7.0
pandas>=2.0.0
selenium>=4.0.0
beautifulsoup4>=4.12.0
webdriver-manager>=4.0.0
```

### 🔗 Related Projects

- Data Analysis: Coming soon
- Price Prediction Model: Coming soon
- Web Dashboard: Coming soon
