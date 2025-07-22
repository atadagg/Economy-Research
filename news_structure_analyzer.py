"""
News Structure Analyzer
Identifies different sections of Turkish news broadcasts: intro, news segments, ads, outro
"""

import re
from openai import OpenAI
from config import OPENAI_API_KEY

class NewsStructureAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Turkish TV patterns for structure identification
        self.intro_patterns = [
            r'(merhaba|iyi akşamlar|günaydın|hoş geldiniz)',
            r'(ana haber|haber bülteni|gece ajansı)',
            r'(sunucu|spiker).*ile',
            r'(başlıyoruz|başlayalım|geçelim)'
        ]
        
        self.ad_patterns = [
            r'(reklam|tanıtım|sponsor)',
            r'(aradan sonra|kısa aradan|moladan sonra)',
            r'(devam ediyor|devam ediyoruz)',
            r'(sizlerle birlikte|beraber)',
            r'(müzik|jenerik)'
        ]
        
        self.outro_patterns = [
            r'(hoşça kalın|iyi geceler|görüşmek üzere)',
            r'(son olarak|bitirirken|sonunda)',
            r'(takip etmeyi|izlemeye devam)',
            r'(yarın|gelecek|bir sonraki)'
        ]
    
    def analyze_structure(self, text):
        """
        Analyze the structure of a news broadcast and identify sections
        Returns: dict with section types and their boundaries
        """
        try:
            # Use AI to identify the overall structure
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this Turkish news broadcast transcript and identify the main structural sections.

Identify these section types and their approximate boundaries:
1. INTRO - Opening, presenter introduction, program start
2. NEWS - Actual news content (there may be multiple news stories)
3. ADS - Commercial breaks, sponsor messages, transitions
4. OUTRO - Closing remarks, goodbye, end of program
5. TRANSITION - Brief transitions between news stories

For each section, provide:
- Section type
- Starting sentence (first few words)
- Brief description

Format your response as:
SECTION_TYPE: "Starting sentence..." - Description

Only identify clear, distinct sections. If the text is mostly one continuous news segment, just mark it as NEWS.

Text (first 2000 chars): {text[:2000]}

Full text length: {len(text)} characters"""
                }],
                temperature=0.1,
                max_tokens=500
            )
            
            ai_analysis = response.choices[0].message.content.strip()
            print(f"AI Structure Analysis: {ai_analysis[:200]}...")
            
            # Parse AI response and create boundaries
            sections = self.parse_ai_structure_response(ai_analysis, text)
            
            # Enhance with pattern-based detection
            sections = self.enhance_with_patterns(text, sections)
            
            return sections
            
        except Exception as e:
            print(f"AI structure analysis failed: {e}")
            return self.fallback_structure_detection(text)
    
    def parse_ai_structure_response(self, ai_response, full_text):
        """Parse the AI response and map to actual text boundaries"""
        sections = []
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            try:
                section_type, description = line.split(':', 1)
                section_type = section_type.strip().upper()
                description = description.strip()
                
                # Extract starting sentence from description
                if '"' in description:
                    start_sentence = description.split('"')[1] if description.count('"') >= 2 else ""
                else:
                    # Try to extract first few words
                    words = description.split()[:5]
                    start_sentence = ' '.join(words)
                
                if start_sentence and len(start_sentence) > 5:
                    # Find this sentence in the full text
                    position = self.find_sentence_position(full_text, start_sentence)
                    if position >= 0:
                        sections.append({
                            'type': section_type,
                            'start_pos': position,
                            'start_sentence': start_sentence,
                            'description': description
                        })
                        
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                continue
        
        # Sort by position and add end positions
        sections.sort(key=lambda x: x['start_pos'])
        for i in range(len(sections)):
            if i < len(sections) - 1:
                sections[i]['end_pos'] = sections[i + 1]['start_pos']
            else:
                sections[i]['end_pos'] = len(full_text)
        
        return sections
    
    def find_sentence_position(self, text, target_sentence):
        """Find the position of a target sentence in the text"""
        # Clean and normalize for searching
        target_clean = re.sub(r'[^\w\s]', ' ', target_sentence.lower()).strip()
        target_words = target_clean.split()[:3]  # Use first 3 words for matching
        
        if len(target_words) < 2:
            return -1
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Find sequence of target words
        for i in range(len(words) - len(target_words) + 1):
            window = words[i:i + len(target_words)]
            if all(target_word in ' '.join(window) for target_word in target_words[:2]):
                # Found approximate match, find sentence boundary
                char_pos = len(' '.join(words[:i]))
                return self.find_sentence_start(text, char_pos)
        
        return -1
    
    def find_sentence_start(self, text, approximate_pos):
        """Find the start of the sentence containing the given position"""
        # Look backward for sentence boundary
        for i in range(approximate_pos, max(0, approximate_pos - 200), -1):
            if i == 0 or (i > 0 and text[i-1] in '.!?\n'):
                # Move forward to first letter
                while i < len(text) and not text[i].isalpha():
                    i += 1
                return i
        
        return max(0, approximate_pos)
    
    def enhance_with_patterns(self, text, ai_sections):
        """Enhance AI-detected sections with pattern-based detection"""
        # If AI found very few sections, add pattern-based detection
        if len(ai_sections) < 2:
            pattern_sections = self.detect_by_patterns(text)
            # Merge with AI sections, preferring AI results
            for pattern_section in pattern_sections:
                # Check if this overlaps with existing AI sections
                overlap = False
                for ai_section in ai_sections:
                    if abs(pattern_section['start_pos'] - ai_section['start_pos']) < 100:
                        overlap = True
                        break
                if not overlap:
                    ai_sections.append(pattern_section)
            
            # Re-sort by position
            ai_sections.sort(key=lambda x: x['start_pos'])
        
        return ai_sections
    
    def detect_by_patterns(self, text):
        """Fallback pattern-based section detection"""
        sections = []
        
        # Find intro patterns
        for pattern in self.intro_patterns:
            matches = list(re.finditer(pattern, text.lower()))
            if matches and matches[0].start() < len(text) * 0.2:  # In first 20%
                sections.append({
                    'type': 'INTRO',
                    'start_pos': matches[0].start(),
                    'description': 'Pattern-detected intro'
                })
                break
        
        # Find ad patterns
        for pattern in self.ad_patterns:
            for match in re.finditer(pattern, text.lower()):
                sections.append({
                    'type': 'ADS',
                    'start_pos': match.start(),
                    'description': 'Pattern-detected ad break'
                })
        
        # Find outro patterns  
        for pattern in self.outro_patterns:
            matches = list(re.finditer(pattern, text.lower()))
            if matches and matches[-1].start() > len(text) * 0.8:  # In last 20%
                sections.append({
                    'type': 'OUTRO',
                    'start_pos': matches[-1].start(),
                    'description': 'Pattern-detected outro'
                })
                break
        
        return sections
    
    def fallback_structure_detection(self, text):
        """Simple fallback if AI analysis fails"""
        sections = []
        
        # Simple heuristic: treat whole text as news unless clear patterns found
        intro_found = any(re.search(pattern, text.lower()) for pattern in self.intro_patterns[:2])
        outro_found = any(re.search(pattern, text.lower()) for pattern in self.outro_patterns[:2])
        
        pos = 0
        if intro_found:
            sections.append({
                'type': 'INTRO',
                'start_pos': 0,
                'end_pos': min(300, len(text) // 10),
                'description': 'Detected intro section'
            })
            pos = sections[-1]['end_pos']
        
        # Main content as news
        end_pos = len(text)
        if outro_found:
            end_pos = max(len(text) - 300, len(text) * 0.9)
        
        sections.append({
            'type': 'NEWS',
            'start_pos': pos,
            'end_pos': int(end_pos),
            'description': 'Main news content'
        })
        
        if outro_found:
            sections.append({
                'type': 'OUTRO',
                'start_pos': int(end_pos),
                'end_pos': len(text),
                'description': 'Detected outro section'
            })
        
        return sections
    
    def extract_sections(self, text, sections):
        """Extract the actual text content for each identified section"""
        extracted = {}
        
        for section in sections:
            start = section['start_pos']
            end = section.get('end_pos', len(text))
            
            section_text = text[start:end].strip()
            section_type = section['type']
            
            # Normalize section types - treat different news-related types as NEWS
            if section_type in ['NEWS', 'TRANSITION', 'SEGMENT']:
                section_type = 'NEWS'
            
            if section_type not in extracted:
                extracted[section_type] = []
            
            extracted[section_type].append({
                'content': section_text,
                'start_pos': start,
                'end_pos': end,
                'description': section.get('description', '')
            })
        
        return extracted