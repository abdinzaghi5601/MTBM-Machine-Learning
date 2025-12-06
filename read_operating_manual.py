"""
Read Operating Manual PDF
==========================

Extracts text from the MTBM operating manual PDF.
"""

import PyPDF2
from pathlib import Path

pdf_path = Path(r"F:\Microtunnelling\M i c r o  -  t u n n e l l i n g\MC\Micro tunellimg\Tbm machine manual\M-1643C\DATEIEN\01_Operating Manual\OperatingManual_M-1675C_C30_V001_EN.pdf")

print("="*80)
print("READING OPERATING MANUAL PDF")
print("="*80)
print(f"\nFile: {pdf_path.name}")
print(f"Size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")

try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        num_pages = len(pdf_reader.pages)
        print(f"Total Pages: {num_pages}")
        
        # Extract text from first 10 pages to get overview
        print("\n" + "="*80)
        print("EXTRACTING TEXT FROM FIRST 10 PAGES (Overview)")
        print("="*80)
        
        for page_num in range(min(10, num_pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            print(f"\n{'='*80}")
            print(f"PAGE {page_num + 1}")
            print(f"{'='*80}")
            print(text[:2000])  # First 2000 chars per page
            if len(text) > 2000:
                print(f"\n... ({len(text) - 2000} more characters)")
        
        # Extract table of contents or key sections
        print("\n\n" + "="*80)
        print("SEARCHING FOR KEY SECTIONS")
        print("="*80)
        
        # Look for steering-related content
        steering_keywords = ['steering', 'cylinder', 'pitch', 'yaw', 'deviation', 
                           'correction', 'alignment', 'laser', 'target']
        
        found_sections = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text().lower()
            
            matches = [kw for kw in steering_keywords if kw in text]
            if matches:
                found_sections.append({
                    'page': page_num + 1,
                    'keywords': matches,
                    'preview': page.extract_text()[:500]
                })
        
        if found_sections:
            print(f"\nFound {len(found_sections)} pages with steering-related content:")
            for section in found_sections[:20]:  # Show first 20 matches
                print(f"\nPage {section['page']}:")
                print(f"  Keywords: {', '.join(section['keywords'])}")
                print(f"  Preview: {section['preview'][:200]}...")
        
        # Extract full text to file (optional, for large PDFs)
        print("\n\n" + "="*80)
        print("EXTRACTING FULL TEXT")
        print("="*80)
        print("This may take a moment for large PDFs...")
        
        full_text = []
        for page_num in range(num_pages):
            if page_num % 50 == 0:
                print(f"  Processing page {page_num + 1}/{num_pages}...")
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                full_text.append(f"\n{'='*80}\nPAGE {page_num + 1}\n{'='*80}\n{text}")
        
        # Save to text file
        output_file = Path("OperatingManual_M-1675C_extracted.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("".join(full_text))
        
        print(f"\n✅ Full text extracted and saved to: {output_file}")
        print(f"   Total pages extracted: {num_pages}")
        print(f"   Total characters: {sum(len(t) for t in full_text):,}")

except FileNotFoundError:
    print(f"❌ Error: File not found at {pdf_path}")
except Exception as e:
    print(f"❌ Error reading PDF: {e}")
    import traceback
    traceback.print_exc()

