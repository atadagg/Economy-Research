"""
Turkish News Economic Content Extractor
Simple tool to extract economic content from Turkish news transcriptions
"""

import os
import re
import json
import sys
from openai import OpenAI
from config import OPENAI_API_KEY, ECONOMIC_KEYWORDS, SEGMENT_MIN_LENGTH, SEGMENT_MAX_LENGTH, ECONOMIC_THRESHOLD

class NewsProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def segment_text(self, text):
        """Split text by news topics, not just word count"""
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Try to detect topic changes with AI first
        segments = self.detect_news_boundaries(text)
        if segments:
            return segments
        
        # Fallback: look for natural topic transitions in Turkish news
        segments = self.segment_by_topic_patterns(text)
        return [s for s in segments if len(s.split()) >= SEGMENT_MIN_LENGTH]
    
    def detect_news_boundaries(self, text):
        """Use AI to detect where different news stories begin/end"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"Analyze this Turkish news transcript. Does it discuss ONE main topic/news story (even with different guests) or MULTIPLE different news stories?\n\nRespond with only:\n- 'SINGLE_TOPIC' if it's all about one main story\n- 'MULTIPLE_TOPICS' if there are clearly different news stories\n\nText: {text[:1500]}"
                }],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            print(f"AI topic analysis: {result}")  # Debug output
            
            if "SINGLE_TOPIC" in result:
                return [text]  # Keep as one segment
            else:
                return None  # Let pattern detection handle it
            
        except Exception as e:
            print(f"AI topic detection failed: {e}")
        
        return None
    
    def segment_by_topic_patterns(self, text):
        """Segment by detecting Turkish news topic transition patterns"""
        # Only look for strong indicators of topic change, not guest changes within same topic
        topic_indicators = [
            # Strong topic transitions
            r'\b(?:Başka|Diğer)\s+(?:bir\s+)?(?:haber|konu|başlık|gelişme)',
            r'\b(?:Gündemi|Ajanda).*?değiştir',
            r'\bBir(?:de|di)?\s+(?:diğer|başka)\s+(?:haber|konu)',
            # Clear location-based news changes
            r'\b(?:İstanbul|Ankara|İzmir|Bursa)\'?(?:da|dan|a)\s+(?:yaşanan|meydana\s+gelen)\s+(?:olay|gelişme)',
            # Show segment transitions (strong indicators)
            r'\b(?:Gece\s+Ajansı|Ana\s+Haber|Özel\s+Haber)\s*-',
            # Very clear topic shift phrases
            r'\b(?:Şimdi\s+de\s+başka\s+bir\s+konuya|Diğer\s+gündem\s+maddesi)'
        ]
        
        segments = []
        current_segment = ""
        words = text.split()
        
        i = 0
        while i < len(words):
            # Check for topic transition patterns
            window = ' '.join(words[i:i+10])  # Look ahead 10 words
            found_transition = False
            
            for pattern in topic_indicators:
                if re.search(pattern, window, re.IGNORECASE):
                    # Found a topic transition
                    if current_segment.strip() and len(current_segment.split()) >= SEGMENT_MIN_LENGTH:
                        segments.append(current_segment.strip())
                        current_segment = ""
                    found_transition = True
                    break
            
            current_segment += " " + words[i]
            i += 1
            
            # Also split if segment gets too long (fallback)
            if len(current_segment.split()) >= SEGMENT_MAX_LENGTH * 2:
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
        
        # Add final segment
        if current_segment.strip():
            if segments and len(current_segment.split()) < SEGMENT_MIN_LENGTH:
                # Merge short final segment with previous
                segments[-1] += " " + current_segment.strip()
            else:
                segments.append(current_segment.strip())
        
        # If no meaningful splits found, return as single segment
        if len(segments) <= 1:
            return [text.strip()]
        
        return segments
    
    def has_economic_keywords(self, text):
        """Quick keyword check"""
        return any(keyword in text.lower() for keyword in ECONOMIC_KEYWORDS)
    
    def classify_economic_content(self, text):
        """Use OpenAI to classify if text contains economic content"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user", 
                    "content": f"Is this Turkish news text about economics/finance? Reply with just 'YES' or 'NO' and a confidence score 0-1:\n\n{text[:1000]}"
                }],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            is_economic = "YES" in result.upper()
            
            # Extract confidence score if present
            confidence = 0.7  # default
            for token in result.split():
                try:
                    if '.' in token:
                        confidence = float(token)
                        break
                except:
                    continue
            
            return is_economic, confidence
            
        except Exception as e:
            print(f"API error: {e}")
            # Fallback to keyword matching
            keywords_found = sum(1 for k in ECONOMIC_KEYWORDS if k in text.lower())
            confidence = min(keywords_found / 5.0, 1.0)
            return confidence > 0.3, confidence
    
    def process_file(self, file_path):
        """Process a single news file"""
        print(f"Processing: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        segments = self.segment_text(content)
        economic_segments = []
        
        for segment in segments:
            if not self.has_economic_keywords(segment):
                continue
                
            is_economic, confidence = self.classify_economic_content(segment)
            
            if is_economic and confidence >= ECONOMIC_THRESHOLD:
                economic_segments.append({
                    'content': segment,
                    'confidence': confidence
                })
        
        return economic_segments
    
    def save_results(self, file_path, segments):
        """Save economic segments to file"""
        if not segments:
            return
        
        os.makedirs("processed", exist_ok=True)
        basename = os.path.basename(file_path).replace('.txt', '')
        output_file = f"processed/{basename}_economic.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Economic content from: {os.path.basename(file_path)}\n")
            f.write(f"# Found {len(segments)} economic segments\n\n")
            
            for i, segment in enumerate(segments):
                f.write(f"## Segment {i+1} (Confidence: {segment['confidence']:.2f})\n\n")
                f.write(segment['content'])
                f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"  → Saved {len(segments)} segments to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <file_path>          # Process single file")
        print(f"  {sys.argv[0]} news/AHaber/         # Process directory")  
        print(f"  {sys.argv[0]} all                  # Process all news files")
        print(f"  {sys.argv[0]} <target> --force     # Force reprocess existing files")
        return
    
    processor = NewsProcessor()
    target = sys.argv[1]
    force_reprocess = "--force" in sys.argv
    
    def should_process_file(file_path):
        """Check if file should be processed (not already done or force flag set)"""
        if force_reprocess:
            return True
        
        basename = os.path.basename(file_path).replace('.txt', '')
        output_file = f"processed/{basename}_economic.txt"
        return not os.path.exists(output_file)
    
    if target == "all":
        # Process all files
        for channel in ["AHaber", "ATV", "Halk"]:
            channel_dir = f"news/{channel}"
            if os.path.exists(channel_dir):
                files = [f for f in os.listdir(channel_dir) if f.endswith('.txt')]
                files_to_process = [f for f in files if should_process_file(os.path.join(channel_dir, f))]
                skipped_count = len(files) - len(files_to_process)
                
                print(f"\n{channel}: {len(files_to_process)} to process, {skipped_count} already done")
                
                for filename in files_to_process:
                    file_path = os.path.join(channel_dir, filename)
                    segments = processor.process_file(file_path)
                    processor.save_results(file_path, segments)
    
    elif os.path.isdir(target):
        # Process directory
        files = [f for f in os.listdir(target) if f.endswith('.txt')]
        files_to_process = [f for f in files if should_process_file(os.path.join(target, f))]
        skipped_count = len(files) - len(files_to_process)
        
        print(f"Directory: {len(files_to_process)} to process, {skipped_count} already done")
        
        for filename in files_to_process:
            file_path = os.path.join(target, filename)
            segments = processor.process_file(file_path)
            processor.save_results(file_path, segments)
    
    elif os.path.isfile(target):
        # Process single file
        if should_process_file(target):
            segments = processor.process_file(target)
            processor.save_results(target, segments)
            
            # Show preview
            if segments:
                print(f"\nPreview of first segment:")
                print(f"Confidence: {segments[0]['confidence']:.2f}")
                print(f"Content: {segments[0]['content'][:200]}...")
        else:
            basename = os.path.basename(target).replace('.txt', '')
            output_file = f"processed/{basename}_economic.txt"
            print(f"File already processed: {output_file}")
            print("Use --force flag to reprocess")
    
    else:
        print(f"Error: {target} not found")

if __name__ == "__main__":
    main() 