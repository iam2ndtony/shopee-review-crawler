import json
import re
import os
import time
import sys

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Installing playwright dependency...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.sync_api import sync_playwright

import urllib.parse
import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')

def get_7_words_filename(url):
    unquoted = urllib.parse.unquote(url)
    match = re.search(r'shopee\.vn/([^/]+)-i\.\d+\.\d+', unquoted)
    raw_slug = match.group(1) if match else unquoted.split('/')[-1].split('?')[0]
    words = re.findall(r'\w+', raw_slug.replace('-', ' '))
    first_7 = words[:7]
    if not first_7:
        first_7 = ["shopee", "product", "reviews"]
    return remove_accents('_'.join(first_7))

def extract_ids_from_url(url):
    match = re.search(r'i\.(\d+)\.(\d+)', url)
    if match:
        return match.group(1), match.group(2)
    numbers = re.findall(r'\d+', url)
    if len(numbers) >= 2:
        return numbers[-2], numbers[-1]
    raise ValueError("Could not extract shop_id and item_id.")

def crawl_shopee_reviews_persistent(product_url, max_reviews=500):
    try:
        shop_id, item_id = extract_ids_from_url(product_url)
        print(f"Extracted Shop ID: {shop_id}, Item ID: {item_id}")
    except ValueError as e:
        print(e)
        return []

    # Dedicated folder to store login session (cookies, state) permanently
    base_dir = os.path.dirname(os.path.abspath(__file__))
    profile_dir = os.path.join(base_dir, "shopee_chrome_profile")
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
        print(f"Created persistent session profile directory: {profile_dir}")

    all_ratings = []
    
    print("\nLaunching browser with persistent session...")
    with sync_playwright() as p:
        # Use Playwright's default Chromium to avoid conflict with your open Google Chrome
        context = p.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False,
            viewport={"width": 1280, "height": 800},
            ignore_default_args=["--enable-automation"],
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ]
        )
        page = context.new_page()
        
        print("Navigating to product page...")
        page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
        
        print("\n>>> BROWSER WINDOW IS OPEN! <<<")
        print("1. In the Chrome window, make sure you are logged in to your Shopee account.")
        print("   - If not logged in: Click 'Log In' and scan the QR code with your phone.")
        print("2. Ensure you are on the product page and can see the product reviews.")
        input("3. Once logged in and ready, press ENTER in this terminal to start crawling...\n")
            
        print("Crawling reviews...")
        limit = 50
        for offset in range(0, max_reviews, limit):
            api_url = f"https://shopee.vn/api/v2/item/get_ratings?filter=0&limit={limit}&offset={offset}&itemid={item_id}&shopid={shop_id}&type=0"
            print(f"Fetching reviews {offset} to {offset + limit}...")
            
            try:
                response_json = page.evaluate("""async (url) => {
                    try {
                        const res = await fetch(url);
                        if (!res.ok) return { _fetch_error: `HTTP ${res.status}` };
                        return await res.json();
                    } catch (e) {
                        return { _fetch_error: e.message };
                    }
                }""", api_url)
                
                if "_fetch_error" in response_json:
                    print(f"Fetch error at offset {offset}: {response_json['_fetch_error']}")
                    break
                    
                if response_json.get("error") != 0 and response_json.get("error") is not None:
                    print(f"Shopee API error code at offset {offset}: {response_json.get('error')} - {response_json.get('error_msg')}")
                    break
                    
                ratings = response_json.get('data', {}).get('ratings', [])
                if not ratings:
                    print(f"No more reviews found at offset {offset}. Stopping.")
                    break
                    
                all_ratings.extend(ratings)
                print(f"Successfully retrieved {len(ratings)} reviews.")
                
            except Exception as e:
                print(f"Failed to fetch offset {offset}: {e}")
                break
                
            time.sleep(1)
            
        context.close()
        
    print(f"\nDone! Crawled {len(all_ratings)} reviews.")
    return all_ratings

if __name__ == "__main__":
    import argparse
    default_url = "https://shopee.vn/H%E1%BB%99p-%C4%91%E1%BB%B1ng-bao-ng%C3%B3n-tay-c%E1%BA%A5t-gi%E1%BB%AF-bao-ngon-tay-ti%E1%BB%87n-d%E1%BB%A5ng-i.133655613.18749577523?extraParams=%7B%22display_model_id%22%3A49676749828%2C%22model_selection_logic%22%3A3%7D"
    
    parser = argparse.ArgumentParser(description="Shopee Review Crawler")
    parser.add_argument("--url", type=str, default=None, help="Shopee product URL")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of reviews to crawl")
    
    args = parser.parse_args()
    
    target_url = args.url
    if not target_url:
        print("\n============================================================")
        print("🔗 NHẬP LINK SẢN PHẨM SHOPEE CẦN CÀO ĐÁNH GIÁ:")
        print(f"Mặc định: {default_url}")
        print("============================================================")
        input_url = input("Dán link vào đây (hoặc nhấn ENTER để dùng link mặc định): ").strip()
        target_url = input_url if input_url else default_url
        
    target_max = args.max
    if not target_max:
        input_max = input("Nhập số lượng review tối đa cần cào (nhấn ENTER để mặc định 1000): ").strip()
        target_max = int(input_max) if input_max.isdigit() else 1000
        
    print(f"\n🚀 Đang chuẩn bị cào:")
    print(f" - URL: {target_url}")
    print(f" - Max reviews: {target_max}")
    
    reviews = crawl_shopee_reviews_persistent(target_url, max_reviews=target_max)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw_reviews")
    os.makedirs(raw_dir, exist_ok=True)
    
    file_prefix = get_7_words_filename(target_url)
    output_file = os.path.join(raw_dir, f"{file_prefix}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(reviews)} reviews to {os.path.abspath(output_file)}")
