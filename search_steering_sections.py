"""
Search for steering-related sections in the extracted manual
"""

import re
from pathlib import Path

text_file = Path("OperatingManual_M-1675C_extracted.txt")

if not text_file.exists():
    print("❌ Extracted text file not found. Please run read_operating_manual.py first.")
    exit(1)

print("="*80)
print("SEARCHING FOR STEERING-RELATED SECTIONS")
print("="*80)

with open(text_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split into pages
pages = content.split("="*80)

# Keywords to search for
steering_keywords = [
    'steering cylinder',
    'steering control',
    'pitch',
    'yaw',
    'deviation',
    'correction',
    'alignment',
    'laser target',
    'steering head',
    'cylinder position',
    'steering rate'
]

print(f"\nTotal pages: {len(pages)}")
print(f"\nSearching for steering-related content...\n")

relevant_pages = []

for i, page in enumerate(pages):
    page_lower = page.lower()
    matches = []
    
    for keyword in steering_keywords:
        if keyword.lower() in page_lower:
            matches.append(keyword)
    
    if matches:
        # Extract page number
        page_match = re.search(r'PAGE (\d+)', page)
        page_num = page_match.group(1) if page_match else f"Unknown ({i})"
        
        # Extract section title if available
        title_match = re.search(r'([A-Z][A-Z\s\-]+)\n', page[:500])
        title = title_match.group(1).strip() if title_match else "No title"
        
        relevant_pages.append({
            'page_num': page_num,
            'title': title,
            'matches': matches,
            'content': page[:1000]  # First 1000 chars
        })

print(f"Found {len(relevant_pages)} pages with steering-related content\n")

# Show most relevant pages
for page_info in relevant_pages[:15]:  # Show top 15
    print("="*80)
    print(f"PAGE {page_info['page_num']} - {page_info['title']}")
    print("="*80)
    print(f"Keywords found: {', '.join(set(page_info['matches']))}")
    print("\nContent preview:")
    print(page_info['content'][:800])
    print("\n...\n")

# Search for specific sections
print("\n" + "="*80)
print("SEARCHING FOR SPECIFIC STEERING OPERATIONS")
print("="*80)

# Look for operation instructions
operation_patterns = [
    r'steering.*operation',
    r'how.*steer',
    r'steering.*procedure',
    r'cylinder.*control',
    r'steering.*calculation'
]

for pattern in operation_patterns:
    matches = re.finditer(pattern, content, re.IGNORECASE)
    for match in list(matches)[:3]:  # First 3 matches
        start = max(0, match.start() - 200)
        end = min(len(content), match.end() + 500)
        context = content[start:end]
        print(f"\nFound: '{match.group()}'")
        print(context)
        print("\n" + "-"*80)

