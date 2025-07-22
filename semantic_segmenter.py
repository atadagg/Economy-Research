"""
Semantic News Segmenter
Intelligently segments news content into coherent stories/topics using semantic understanding
"""

import re
import nltk
from openai import OpenAI
from config import OPENAI_API_KEY

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticSegmenter:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Turkish news transition markers
        self.transition_markers = [
            r'(diğer|başka|bir diğer) (haber|konu|gelişme)',
            r'(şimdi|şu anda|bunun yanında) .*',
            r'(geçelim|bakalım) .*',
            r'(öte yandan|bu arada|ayrıca)',
            r'(son dakika|flaş|acil)',
            r'(gündeme|ajandaya|haberlere) (geç|dön)',
        ]
        
        # Topic change indicators
        self.topic_change_patterns = [
            r'(farklı|bambaşka|yeni) (konu|mesele)',
            r'(politik|siyasi|askeri|ekonomik|sosyal) (alan|konu)',
            r'(iç|dış|yerel|uluslararası) (politika|haber)',
            r'(spor|kültür|sanat|eğitim|sağlık) (haberleri|alanında)'
        ]
    
    def segment_news_content(self, news_sections):
        """
        Segment news content into coherent stories
        Args: news_sections - list of news section dictionaries from structure analyzer
        Returns: list of individual news story segments
        """
        all_segments = []
        
        for section in news_sections:
            content = section['content']
            
            # Skip very short sections
            if len(content.split()) < 50:
                continue
            
            # Get semantic segments for this section
            segments = self.semantic_segment(content)
            
            # Add metadata
            for i, segment in enumerate(segments):
                segment.update({
                    'section_start_pos': section['start_pos'],
                    'section_description': section.get('description', ''),
                    'segment_id': f"{section['start_pos']}_{i}"
                })
            
            all_segments.extend(segments)
        
        return all_segments
    
    def semantic_segment(self, text):
        """
        Segment text into coherent news stories using semantic understanding
        """
        try:
            # First, try AI-based segmentation
            ai_segments = self.ai_semantic_segmentation(text)
            
            if ai_segments and len(ai_segments) > 1:
                # Validate AI segments
                validated_segments = self.validate_segments(ai_segments, text)
                if validated_segments:
                    return validated_segments
            
            # Fallback to pattern-based + sentence-level analysis
            return self.hybrid_segmentation(text)
            
        except Exception as e:
            print(f"Semantic segmentation failed: {e}")
            return self.simple_paragraph_segmentation(text)
    
    def ai_semantic_segmentation(self, text):
        """
        Use AI to identify semantic boundaries in news content
        """
        # For very long texts, process in chunks
        if len(text) > 3000:
            return self.chunk_and_segment(text)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Analyze this Turkish news content and identify distinct news stories or major topic changes.

Instructions:
1. Look for places where the content shifts to a completely different news story or major topic
2. DO NOT split on minor subtopics, guest changes, or different perspectives on the same story
3. Only identify clear boundaries where a new, distinct news story begins
4. Each segment should be a complete, coherent news story

If there are multiple distinct stories, provide the starting sentence of each NEW story (not the first one):

Format: 
BOUNDARY: "Exact starting sentence of story 2"
BOUNDARY: "Exact starting sentence of story 3"
...

If it's all one continuous story/topic, respond: "SINGLE_STORY"

Text: {text[:2500]}"""
            }],
            temperature=0.1,
            max_tokens=400
        )
        
        result = response.choices[0].message.content.strip()
        print(f"AI segmentation result: {result[:150]}...")
        
        if "SINGLE_STORY" in result:
            return [{'content': text, 'confidence': 0.9, 'method': 'ai_single'}]
        
        # Parse boundaries
        boundaries = []
        for line in result.split('\n'):
            if line.strip().startswith('BOUNDARY:') and '"' in line:
                try:
                    boundary_sentence = line.split('"')[1]
                    if len(boundary_sentence) > 10:  # Valid boundary
                        boundaries.append(boundary_sentence)
                except:
                    continue
        
        if not boundaries:
            return [{'content': text, 'confidence': 0.8, 'method': 'ai_single_fallback'}]
        
        # Split by boundaries
        return self.split_by_semantic_boundaries(text, boundaries)
    
    def chunk_and_segment(self, text):
        """
        Handle very long texts by processing in chunks
        """
        words = text.split()
        chunk_size = 500  # words per chunk
        overlap = 50     # word overlap between chunks
        
        chunks = []
        segments = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) < 100:  # Skip tiny chunks
                continue
            chunks.append({'content': chunk, 'start_word': i})
        
        # Process each chunk
        for chunk in chunks:
            chunk_segments = self.ai_semantic_segmentation(chunk['content'])
            
            # Adjust positions for full text
            for segment in chunk_segments:
                segment['chunk_start'] = chunk['start_word']
            
            segments.extend(chunk_segments)
        
        # Merge overlapping segments and resolve boundaries
        return self.merge_chunk_segments(segments, text)
    
    def split_by_semantic_boundaries(self, text, boundaries):
        """
        Split text at the identified semantic boundaries
        """
        segments = []
        current_pos = 0
        
        sentences = self.split_into_sentences(text)
        sentence_positions = self.get_sentence_positions(text, sentences)
        
        for boundary in boundaries:
            # Find where this boundary occurs
            boundary_pos = self.find_boundary_in_sentences(sentences, sentence_positions, boundary)
            
            if boundary_pos > current_pos:
                # Create segment
                start_char = sentence_positions[current_pos]['start'] if current_pos < len(sentence_positions) else 0
                end_char = sentence_positions[boundary_pos]['start'] if boundary_pos < len(sentence_positions) else len(text)
                
                segment_text = text[start_char:end_char].strip()
                if len(segment_text.split()) >= 30:  # Minimum viable segment
                    segments.append({
                        'content': segment_text,
                        'start_pos': start_char,
                        'end_pos': end_char,
                        'confidence': 0.85,
                        'method': 'ai_boundary'
                    })
                
                current_pos = boundary_pos
        
        # Add final segment
        if current_pos < len(sentences):
            start_char = sentence_positions[current_pos]['start']
            segment_text = text[start_char:].strip()
            if len(segment_text.split()) >= 30:
                segments.append({
                    'content': segment_text,
                    'start_pos': start_char,
                    'end_pos': len(text),
                    'confidence': 0.85,
                    'method': 'ai_boundary_final'
                })
        
        return segments if segments else [{'content': text, 'confidence': 0.7, 'method': 'ai_fallback'}]
    
    def split_into_sentences(self, text):
        """
        Split text into sentences using NLTK
        """
        try:
            sentences = nltk.sent_tokenize(text, language='turkish')
            return sentences
        except:
            # Fallback simple splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def get_sentence_positions(self, text, sentences):
        """
        Get character positions of each sentence in the original text
        """
        positions = []
        current_pos = 0
        
        for sentence in sentences:
            # Find this sentence in the text
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
                
            # Look for the sentence starting from current position
            start_pos = text.find(sentence_clean, current_pos)
            if start_pos == -1:
                # Try fuzzy matching with first few words
                words = sentence_clean.split()[:3]
                search_text = ' '.join(words)
                start_pos = text.find(search_text, current_pos)
            
            if start_pos >= 0:
                positions.append({
                    'sentence': sentence_clean,
                    'start': start_pos,
                    'end': start_pos + len(sentence_clean)
                })
                current_pos = start_pos + len(sentence_clean)
            
        return positions
    
    def find_boundary_in_sentences(self, sentences, positions, boundary_text):
        """
        Find which sentence index corresponds to the boundary text
        """
        boundary_words = boundary_text.lower().split()[:4]  # First 4 words
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            sentence_words = sentence_lower.split()[:4]
            
            # Check if boundary words match beginning of this sentence
            matches = sum(1 for bw in boundary_words if bw in sentence_words)
            if matches >= len(boundary_words) * 0.75:  # 75% match threshold
                return i
        
        return -1
    
    def hybrid_segmentation(self, text):
        """
        Combine pattern-based detection with sentence analysis
        """
        segments = []
        
        # First try pattern-based splitting
        pattern_splits = self.pattern_based_split(text)
        
        if len(pattern_splits) > 1:
            # Validate each pattern split with semantic analysis
            for split in pattern_splits:
                if len(split['content'].split()) >= 50:
                    segments.append({
                        **split,
                        'confidence': 0.75,
                        'method': 'pattern_validated'
                    })
        else:
            # No clear patterns, try paragraph-based segmentation
            return self.smart_paragraph_segmentation(text)
        
        return segments if segments else [{'content': text, 'confidence': 0.6, 'method': 'hybrid_fallback'}]
    
    def pattern_based_split(self, text):
        """
        Split based on Turkish news transition patterns
        """
        splits = []
        current_pos = 0
        
        # Look for transition markers
        for pattern in self.transition_markers + self.topic_change_patterns:
            for match in re.finditer(pattern, text.lower()):
                match_pos = match.start()
                
                # Find sentence boundary
                sentence_start = self.find_sentence_start_before(text, match_pos)
                
                if sentence_start > current_pos + 200:  # Minimum segment size
                    # Create segment
                    segment_text = text[current_pos:sentence_start].strip()
                    if len(segment_text.split()) >= 30:
                        splits.append({
                            'content': segment_text,
                            'start_pos': current_pos,
                            'end_pos': sentence_start,
                            'trigger_pattern': pattern
                        })
                        current_pos = sentence_start
        
        # Add final segment
        if current_pos < len(text) - 200:
            final_text = text[current_pos:].strip()
            if len(final_text.split()) >= 30:
                splits.append({
                    'content': final_text,
                    'start_pos': current_pos,
                    'end_pos': len(text)
                })
        
        return splits
    
    def find_sentence_start_before(self, text, pos):
        """
        Find the start of the sentence that contains or precedes the given position
        """
        # Look backward for sentence boundary
        for i in range(pos, max(0, pos - 100), -1):
            if i == 0 or text[i-1] in '.!?\n':
                # Move forward to first letter
                while i < len(text) and not text[i].isalpha():
                    i += 1
                return i
        return pos
    
    def smart_paragraph_segmentation(self, text):
        """
        Segment based on paragraph breaks and content analysis
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) <= 1:
            # No paragraph breaks, try other delimiters
            paragraphs = self.split_by_content_markers(text)
        
        segments = []
        current_segment = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para.split()) < 10:
                continue
            
            # Check if this paragraph starts a new topic
            if self.is_topic_start(para) and current_segment and len(current_segment.split()) >= 50:
                # Save current segment
                segments.append({
                    'content': current_segment.strip(),
                    'confidence': 0.7,
                    'method': 'paragraph_smart'
                })
                current_segment = para
            else:
                # Add to current segment
                current_segment += (" " + para) if current_segment else para
        
        # Add final segment
        if current_segment.strip() and len(current_segment.split()) >= 30:
            segments.append({
                'content': current_segment.strip(),
                'confidence': 0.7,
                'method': 'paragraph_smart_final'
            })
        
        return segments if segments else [{'content': text, 'confidence': 0.6, 'method': 'paragraph_fallback'}]
    
    def is_topic_start(self, paragraph):
        """
        Determine if a paragraph starts a new topic
        """
        para_lower = paragraph.lower().strip()
        
        # Check for topic start indicators
        topic_starts = [
            r'^(şimdi|şu anda|bunun yanında)',
            r'^(öte yandan|bu arada|ayrıca)',
            r'^(diğer|başka|bir diğer)',
            r'^(geçelim|bakalım)',
            r'^[A-Z][a-z]+ (şehri|ili|bölgesi)',  # Location mentions
            r'^\d+\.',  # Numbered items
        ]
        
        return any(re.match(pattern, para_lower) for pattern in topic_starts)
    
    def split_by_content_markers(self, text):
        """
        Split by content markers when no paragraph breaks exist
        """
        # Split by speaker changes, timestamps, etc.
        markers = [
            r'\d{2}:\d{2}',  # Time stamps
            r'[A-Z][A-Z ]{10,}:',  # Speaker names in caps
            r'\[.*?\]',  # Bracketed content
        ]
        
        splits = [text]  # Start with whole text
        
        for marker in markers:
            new_splits = []
            for split in splits:
                parts = re.split(marker, split)
                new_splits.extend([p.strip() for p in parts if p.strip()])
            splits = new_splits
        
        return [s for s in splits if len(s.split()) >= 20]
    
    def simple_paragraph_segmentation(self, text):
        """
        Simple fallback segmentation
        """
        # Just split by major punctuation and length
        sentences = self.split_into_sentences(text)
        
        segments = []
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            # If adding this sentence would make segment too long, or if we hit natural break
            if (current_length + sentence_length > 200 and current_length >= 50) or \
               (current_length > 0 and self.is_segment_boundary(sentence)):
                
                # Save current segment
                segment_text = ' '.join(current_segment).strip()
                if len(segment_text.split()) >= 30:
                    segments.append({
                        'content': segment_text,
                        'confidence': 0.5,
                        'method': 'simple_sentence'
                    })
                
                # Start new segment
                current_segment = [sentence]
                current_length = sentence_length
            else:
                current_segment.append(sentence)
                current_length += sentence_length
        
        # Add final segment
        if current_segment:
            segment_text = ' '.join(current_segment).strip()
            if len(segment_text.split()) >= 30:
                segments.append({
                    'content': segment_text,
                    'confidence': 0.5,
                    'method': 'simple_final'
                })
        
        return segments if segments else [{'content': text, 'confidence': 0.4, 'method': 'simple_fallback'}]
    
    def is_segment_boundary(self, sentence):
        """
        Check if sentence indicates a natural segment boundary
        """
        sentence_lower = sentence.lower().strip()
        
        boundary_indicators = [
            r'^(şimdi|şu anda|bunun yanında)',
            r'^(öte yandan|bu arada|ayrıca)',
            r'^(diğer|başka|bir diğer)',
            r'^(son olarak|sonuç olarak|özetlemek gerekirse)',
        ]
        
        return any(re.match(pattern, sentence_lower) for pattern in boundary_indicators)
    
    def validate_segments(self, segments, original_text):
        """
        Validate that segments are coherent and complete
        """
        validated = []
        
        for segment in segments:
            content = segment['content']
            
            # Check minimum length
            if len(content.split()) < 30:
                continue
            
            # Check coherence (starts and ends properly)
            if self.is_coherent_segment(content):
                validated.append(segment)
            
        return validated if validated else None
    
    def is_coherent_segment(self, content):
        """
        Check if segment is coherent (proper start/end, complete thoughts)
        """
        words = content.split()
        if len(words) < 30:
            return False
        
        # Check start - shouldn't start with conjunctions/fragments
        first_words = ' '.join(words[:3]).lower()
        bad_starts = ['ve ', 'ama ', 'fakat ', 'da ', 'de ', 'ki ']
        if any(first_words.startswith(start) for start in bad_starts):
            return False
        
        # Check end - should end with proper punctuation
        last_sentence = content.strip()[-50:]
        if not any(last_sentence.endswith(punct) for punct in ['.', '!', '?', '...']):
            return False
        
        return True
    
    def merge_chunk_segments(self, chunk_segments, full_text):
        """
        Merge segments from chunked processing, handling overlaps
        """
        if not chunk_segments:
            return [{'content': full_text, 'confidence': 0.4, 'method': 'merge_fallback'}]
        
        # For now, simple implementation - just return the segments
        # In production, you'd want to handle overlaps and merge boundaries
        merged = []
        
        for segment in chunk_segments:
            if len(segment['content'].split()) >= 30:
                merged.append({
                    'content': segment['content'],
                    'confidence': segment.get('confidence', 0.6),
                    'method': 'chunk_merged'
                })
        
        return merged if merged else [{'content': full_text, 'confidence': 0.4, 'method': 'merge_fallback'}]