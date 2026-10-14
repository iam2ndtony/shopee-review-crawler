import json
import csv
import os
import re
from datetime import datetime

def clean_comment(comment):
    if not comment:
        return ""
    # Replace multiple spaces/newlines with single ones
    text = re.sub(r'\s+', ' ', comment)
    # Remove common boilerplate templates or format them nicely
    text = re.sub(r'(Chất lượng sản phẩm|Tính năng nổi bật|Đúng với mô tả|Chất liệu|Thiết kế|Công dụng|Mùi hương|Độ bền):\s*', '', text)
    return text.strip()

def is_spam(comment):
    # If comment is empty or very short
    if not comment or len(comment) < 3:
        return True
    
    # Check for repetitive characters (e.g., "hihihihihihi", "aaaaaaa")
    if re.search(r'(.)\1{4,}', comment):
        return True
        
    return False

def clean_reviews(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} does not exist.")
        return
        
    with open(json_file_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return

    # First pass: Clean comments and count frequencies to detect duplicates/bots
    cleaned_records = []
    comment_counts = {}
    
    for item in raw_data:
        comment = item.get("comment", "")
        cleaned_text = clean_comment(comment)
        
        # Skip empty comments after cleaning
        if not cleaned_text:
            continue
            
        comment_counts[cleaned_text] = comment_counts.get(cleaned_text, 0) + 1
        
        # Get variation model name
        variation = ""
        product_items = item.get("product_items", [])
        if product_items and isinstance(product_items, list):
            variation = product_items[0].get("model_name", "")
            
        # Format ctime to YYYY-MM-DD HH:MM:SS
        ctime = item.get("ctime")
        formatted_date = ""
        if ctime:
            try:
                formatted_date = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
                
        # Seller reply
        seller_reply = ""
        reply_obj = item.get("ItemRatingReply")
        if reply_obj and isinstance(reply_obj, dict):
            seller_reply = reply_obj.get("comment", "")

        cleaned_records.append({
            "username": item.get("author_username", "anonymous"),
            "rating": item.get("rating_star", 5),
            "original_comment": comment.strip(),
            "cleaned_comment": cleaned_text,
            "variation": variation,
            "date": formatted_date,
            "images_count": len(item.get("images", []) or []),
            "videos_count": len(item.get("videos", []) or []),
            "like_count": item.get("like_count", 0),
            "seller_reply": seller_reply.strip()
        })

    # Second pass: Apply sentiment label and bot/duplicate detection
    cleaned_data = []
    botted_count = 0
    
    for record in cleaned_records:
        cleaned_text = record["cleaned_comment"]
        rating = record["rating"]
        
        # Sentiment rating mapping
        if rating >= 4:
            sentiment = "positive"
        elif rating == 3:
            sentiment = "neutral"
        else:
            sentiment = "negative"
            
        # Bot detection (duplicate check)
        freq = comment_counts.get(cleaned_text, 1)
        is_potential_bot = False
        if freq > 1:
            # Real users rarely copy-paste long sentences or repeat same comments unless it's a bot
            # Short duplicates like "ok", "tốt" could just be coincidence, so check length or high frequency
            if len(cleaned_text) >= 15 or freq > 2:
                is_potential_bot = True
                
        if is_potential_bot:
            botted_count += 1
            
        record["sentiment"] = sentiment
        record["is_potential_spam"] = is_spam(cleaned_text)
        record["is_potential_bot"] = is_potential_bot
        record["duplicate_count"] = freq
        
        cleaned_data.append(record)

    # Save to CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_dir = os.path.join(base_dir, "cleaned_reviews")
    os.makedirs(cleaned_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    csv_file = os.path.join(cleaned_dir, f"{base_name}_cleaned.csv")
    json_cleaned_file = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")
    
    fields = [
        "username", "rating", "sentiment", "cleaned_comment", "original_comment", 
        "variation", "date", "images_count", "videos_count", "like_count", 
        "seller_reply", "is_potential_spam", "is_potential_bot", "duplicate_count"
    ]
    
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cleaned_data)
        
    # Save to JSON
    with open(json_cleaned_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n--- Data Cleaning Summary ---")
    print(f"Original reviews:   {len(raw_data)}")
    print(f"Cleaned reviews:    {len(cleaned_data)}")
    print(f"Botted/Dup flagged: {botted_count}")
    print(f"Saved cleaned data to:")
    print(f"  CSV:  {os.path.abspath(csv_file)}")
    print(f"  JSON: {os.path.abspath(json_cleaned_file)}")

if __name__ == "__main__":
    import sys
    target_file = None
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw_reviews")
    os.makedirs(raw_dir, exist_ok=True)
    
    if len(sys.argv) >= 2:
        target_file = sys.argv[1]
    else:
        raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json")]
        if not raw_files:
            raw_files = [f for f in os.listdir(base_dir) if f.endswith(".json") and not f.endswith("_cleaned.json")]
            
        if len(raw_files) == 1:
            target_file = raw_files[0]
            print(f"Cleaning file: {target_file}")
        elif len(raw_files) > 1:
            print("Found these JSON review files:")
            for idx, f in enumerate(raw_files):
                print(f"  [{idx}] {os.path.basename(f)}")
            choice = input("\nEnter number to clean (or press ENTER for first file, or type 'all'): ").strip()
            if choice.lower() == "all":
                for f in raw_files:
                    clean_reviews(f)
                sys.exit(0)
            elif choice.isdigit() and int(choice) < len(raw_files):
                target_file = raw_files[int(choice)]
            else:
                target_file = raw_files[0]
        else:
            print("No raw review JSON files found in 'raw_reviews' directory.")
            sys.exit(0)
            
    if target_file:
        clean_reviews(target_file)
