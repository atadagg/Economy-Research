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
        """Split text by news topics using intelligent boundary detection"""
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Use AI to find precise topic boundaries
        segments = self.find_topic_boundaries(text)
        if segments and len(segments) > 1:
            # Validate and refine segments
            segments = self.validate_and_refine_segments(segments)
        else:
            segments = [text]  # Keep as single segment if no clear boundaries
        
        return [s for s in segments if len(s.split()) >= SEGMENT_MIN_LENGTH]
    
    def find_topic_boundaries(self, text):
        """Use AI to find precise boundaries between different news topics"""
        try:
            # First, check if there are multiple topics at all
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this Turkish news transcript carefully. 

Does it contain:
1. ONE main news topic/story (even with different segments, guests, or perspectives on the same story)
2. MULTIPLE distinct news stories/topics

If MULTIPLE topics, identify where each new topic begins by finding the sentence that starts each new story. Look for:
- Topic transitions (moving from one news story to another)
- NOT just guest changes or commercial breaks within the same story
- NOT just different perspectives on the same story

Respond in this format:
- If ONE topic: "SINGLE_TOPIC"
- If MULTIPLE topics: "BOUNDARIES: [sentence where topic 2 starts]|||[sentence where topic 3 starts]|||..."

Only include clear, distinct topic changes.

Text: {text[:2000]}"""
                }],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            print(f"AI boundary analysis: {result[:100]}...")
            
            if "SINGLE_TOPIC" in result:
                return [text]
            
            if "BOUNDARIES:" in result:
                # Extract boundary sentences
                boundaries_part = result.split("BOUNDARIES:")[1].strip()
                boundary_sentences = [s.strip() for s in boundaries_part.split("|||") if s.strip()]
                
                if boundary_sentences:
                    return self.split_by_boundaries(text, boundary_sentences)
            
            return None
            
        except Exception as e:
            print(f"AI boundary detection failed: {e}")
            return None
    
    def split_by_boundaries(self, text, boundary_sentences):
        """Split text at the identified boundary sentences"""
        segments = []
        current_segment = text
        
        for boundary in boundary_sentences:
            # Find where this boundary sentence appears in the current segment
            boundary_clean = re.sub(r'[^\w\s]', '', boundary.lower())
            words = boundary_clean.split()
            
            if len(words) < 3:  # Skip too-short boundaries
                continue
            
            # Look for the boundary in the text (fuzzy matching)
            best_pos = self.find_boundary_position(current_segment, words)
            
            if best_pos > 0:
                # Split at this position
                before = current_segment[:best_pos].strip()
                after = current_segment[best_pos:].strip()
                
                if before and len(before.split()) >= SEGMENT_MIN_LENGTH:
                    segments.append(before)
                
                current_segment = after
        
        # Add the final segment
        if current_segment and len(current_segment.split()) >= SEGMENT_MIN_LENGTH:
            segments.append(current_segment)
        
        return segments if len(segments) > 1 else [text]
    
    def find_boundary_position(self, text, boundary_words):
        """Find the best position in text where boundary words appear"""
        text_lower = text.lower()
        words = text.split()
        
        # Look for sequences of boundary words
        for i in range(len(words) - len(boundary_words) + 1):
            window = words[i:i + len(boundary_words)]
            window_text = ' '.join(window).lower()
            
            # Check for fuzzy match (at least 70% of words match)
            matches = sum(1 for bw in boundary_words if bw in window_text)
            if matches >= len(boundary_words) * 0.7:
                # Find sentence boundary near this position
                char_pos = len(' '.join(words[:i]))
                return self.find_sentence_boundary(text, char_pos)
        
        return -1
    
    def find_sentence_boundary(self, text, approximate_pos):
        """Find the nearest sentence boundary to the given position"""
        # Look backwards for sentence end
        for i in range(approximate_pos, max(0, approximate_pos - 200), -1):
            if i < len(text) and text[i] in '.!?':
                # Move forward to start of next sentence
                for j in range(i + 1, min(len(text), i + 50)):
                    if text[j].isalpha():
                        return j
        
        # If no good boundary found, return approximate position
        return max(0, approximate_pos)
    
    def validate_and_refine_segments(self, segments):
        """Validate that segments are coherent and complete"""
        refined_segments = []
        
        for i, segment in enumerate(segments):
            # Check if segment seems complete and coherent
            is_complete = self.check_segment_completeness(segment)
            
            if is_complete:
                refined_segments.append(segment)
            elif i > 0:
                # If segment seems incomplete, merge with previous
                refined_segments[-1] = refined_segments[-1] + " " + segment
            else:
                # First segment - keep it even if seems incomplete
                refined_segments.append(segment)
        
        return refined_segments
    
    def check_segment_completeness(self, segment):
        """Check if a segment seems like a complete, coherent story"""
        # Basic heuristics for completeness
        words = segment.split()
        
        # Too short segments are likely incomplete
        if len(words) < SEGMENT_MIN_LENGTH:
            return False
        
        # Check if starts and ends reasonably
        first_words = ' '.join(words[:5]).lower()
        last_words = ' '.join(words[-5:]).lower()
        
        # Starts abruptly (likely cut off)
        if any(first_words.startswith(word) for word in ['ve ', 'da ', 'de ', 'ama ', 'fakat ']):
            return False
        
        # Ends abruptly (likely cut off)
        if not any(last_words.endswith(punct) for punct in ['.', '!', '?', '...']):
            return False
        
        return True

    def segment_by_topic_patterns(self, text):
        """DEPRECATED: Keep as fallback only"""
        # This method is now deprecated but kept as ultimate fallback
        return [text]
    

    
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
                    "content": f"""Analyze this Turkish news text. Does it contain ANY economic/financial content including: trade, investment, economy, defense industry, markets, prices, wages, business, finance, economic policy, economic impact of events?

In Turkish media, economic topics are often discussed within political or military contexts. Look for economic implications, business impacts, trade discussions, investment mentions, etc.

Reply with 'YES' if ANY significant economic content is present, 'NO' if purely non-economic, and a confidence score 0-1:

{text[:1500]}"""
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
            else:
                print(f"    → Segment rejected: is_economic={is_economic}, confidence={confidence:.2f}")
        
        return economic_segments
    
    def save_results(self, file_path, segments):
        """Save economic segments to file"""
        if not segments:
            print(f"  → No economic content found")
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
        for channel in ["AHaber", "ATV", "Halk", "KanalD", "Show", "SozcuTV"]:
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