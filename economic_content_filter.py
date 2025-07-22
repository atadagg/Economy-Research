"""
Economic Content Filter
Removes non-economic sentences while preserving context and flow
"""

import re
import nltk
from openai import OpenAI
from config import OPENAI_API_KEY, ECONOMIC_KEYWORDS, ECONOMIC_THRESHOLD

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EconomicContentFilter:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Extended economic keywords for Turkish content
        self.core_economic_terms = ECONOMIC_KEYWORDS + [
            "fiyat artışı", "ücret zammı", "gelir dağılımı", "yoksulluk sınırı",
            "işsizlik oranı", "enflasyon hedefi", "kur korumalı", "tl", "euro",
            "dolar", "sterlin", "çin yuanı", "bitcoin", "kripto", "blockchain",
            "borsa istanbul", "bist", "imkb", "cb", "tcmb", "maliye bakanlığı",
            "hazine", "kamu borcu", "bütçe açığı", "cari açık", "dış ticaret",
            "rezerv", "altın rezervi", "döviz rezervi", "swap", "repo",
            "faiz indirimi", "faiz artışı", "para politikası", "fiscal",
            "monetary", "gdp büyümesi", "reel sektör", "üretim", "sanayi",
            "tarım", "hizmet sektörü", "turizm geliri", "dış yatırım"
        ]
        
        # Context preservation terms - keep sentences that provide context
        self.context_preservation_terms = [
            "açıkladı", "belirtti", "ifade etti", "söyledi", "dedi ki",
            "basın toplantısında", "açıklamasında", "değerlendirmesinde",
            "raporunda", "araştırmasında", "verilerine göre", "istatistikleri",
            "uzmanı", "analisti", "başkanı", "bakanı", "yetkilisi",
            "kurumu", "kurumundan", "bakanlığından", "bankasından"
        ]
        
        # Non-economic topics that should be filtered out completely
        self.non_economic_topics = [
            "spor", "futbol", "basketbol", "voleybol", "tenis", "golf",
            "sanat", "müzik", "sinema", "televizyon", "dizi", "film",
            "sağlık", "hastalık", "tedavi", "aşı", "pandemi", "grip",
            "eğitim", "okul", "üniversite", "öğrenci", "öğretmen",
            "hava durumu", "meteoroloji", "yağmur", "kar", "güneş",
            "trafik", "kaza", "yaralı", "ölü", "hastane", "ambulans",
            "terör", "pkkli", "terörist", "asker", "şehit", "operasyon"
        ]
    
    def filter_economic_content(self, news_segments):
        """
        Filter news segments to keep only economic content while preserving context
        Args: news_segments - list of news segment dictionaries
        Returns: list of filtered segments with economic content only
        """
        filtered_segments = []
        
        for segment in news_segments:
            content = segment['content']
            
            # Quick keyword check - skip if no economic terms
            if not self.has_economic_indicators(content):
                print(f"Segment skipped - no economic indicators")
                continue
            
            # Filter sentences within the segment
            filtered_content = self.filter_sentences_in_segment(content)
            
            if filtered_content and len(filtered_content.split()) >= 30:
                filtered_segment = segment.copy()
                filtered_segment['content'] = filtered_content
                filtered_segment['original_length'] = len(content.split())
                filtered_segment['filtered_length'] = len(filtered_content.split())
                filtered_segment['retention_ratio'] = filtered_segment['filtered_length'] / filtered_segment['original_length']
                
                filtered_segments.append(filtered_segment)
                print(f"Segment filtered: {filtered_segment['original_length']} → {filtered_segment['filtered_length']} words ({filtered_segment['retention_ratio']:.2f})")
            else:
                print(f"Segment rejected after filtering - too short or no content")
        
        return filtered_segments
    
    def has_economic_indicators(self, text):
        """
        Quick check if text contains economic indicators
        """
        text_lower = text.lower()
        
        # Check for core economic terms
        economic_term_count = sum(1 for term in self.core_economic_terms if term in text_lower)
        
        # Check for numeric economic indicators
        has_percentages = bool(re.search(r'\d+[,.]?\d*\s*%', text))
        has_currency = bool(re.search(r'(tl|lira|dolar|euro|sterlin)', text_lower))
        has_numbers_with_economic_context = bool(re.search(r'\d+[,.]?\d*\s*(milyon|milyar|bin|trilyon)', text_lower))
        
        # Threshold: need at least 2 economic terms OR 1 term + numeric indicators
        return economic_term_count >= 2 or (economic_term_count >= 1 and (has_percentages or has_currency or has_numbers_with_economic_context))
    
    def filter_sentences_in_segment(self, content):
        """
        Filter individual sentences within a segment, preserving context
        """
        try:
            # Use AI for intelligent sentence filtering
            ai_filtered = self.ai_sentence_filtering(content)
            
            if ai_filtered and len(ai_filtered.split()) >= len(content.split()) * 0.3:
                return ai_filtered
            
            # Fallback to rule-based filtering
            return self.rule_based_sentence_filtering(content)
            
        except Exception as e:
            print(f"Sentence filtering failed: {e}")
            return self.rule_based_sentence_filtering(content)
    
    def ai_sentence_filtering(self, content):
        """
        Use AI to filter sentences while preserving context and flow
        """
        # For very long content, process in chunks
        if len(content) > 2500:
            return self.chunk_based_filtering(content)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Filter this Turkish news content to keep ONLY sentences related to economics/finance/business, while preserving context and natural flow.

KEEP sentences that discuss:
- Economic data, statistics, indicators
- Financial markets, investments, trading
- Government economic policies
- Business news, company developments
- Prices, inflation, costs, wages
- Banking, credit, monetary policy
- Trade, imports, exports
- Economic analysis or expert opinions
- Context sentences that support economic content

REMOVE sentences about:
- Pure politics without economic impact
- Sports, entertainment, culture
- Weather, traffic, accidents
- Health/medical topics (unless economic impact)
- Military/security (unless economic impact)
- Personal stories without economic relevance

IMPORTANT: Maintain natural flow by keeping transition sentences and context that makes economic content understandable.

Return the filtered text with complete sentences and proper flow:

{content}"""
            }],
            temperature=0.1,
            max_tokens=1500
        )
        
        filtered_content = response.choices[0].message.content.strip()
        
        # Validate the result
        if len(filtered_content.split()) < 20:
            print("AI filtering result too short, using fallback")
            return None
        
        # Check if it still contains economic indicators
        if not self.has_economic_indicators(filtered_content):
            print("AI filtering removed economic indicators, using fallback")
            return None
        
        return filtered_content
    
    def chunk_based_filtering(self, content):
        """
        Filter very long content by processing in chunks
        """
        sentences = self.split_into_sentences(content)
        chunk_size = 15  # sentences per chunk
        overlap = 3      # sentence overlap for context
        
        filtered_chunks = []
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = ' '.join(chunk_sentences)
            
            if len(chunk_text.split()) < 50:
                continue
            
            # Filter this chunk
            filtered_chunk = self.ai_sentence_filtering(chunk_text)
            if filtered_chunk and len(filtered_chunk.split()) >= 20:
                filtered_chunks.append(filtered_chunk)
        
        # Combine chunks, removing overlap
        return self.merge_filtered_chunks(filtered_chunks)
    
    def merge_filtered_chunks(self, chunks):
        """
        Merge filtered chunks while handling overlaps
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Simple merging for now - in production you'd want to handle sentence overlaps
        merged = chunks[0]
        
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            
            # Find potential overlap by checking if first sentence of current chunk
            # is similar to last sentences of merged content
            chunk_sentences = self.split_into_sentences(chunk)
            merged_sentences = self.split_into_sentences(merged)
            
            if chunk_sentences and merged_sentences:
                first_chunk_sentence = chunk_sentences[0]
                
                # Check if this sentence is similar to recent merged sentences
                overlap_found = False
                for j in range(max(0, len(merged_sentences) - 3), len(merged_sentences)):
                    if self.sentences_similar(first_chunk_sentence, merged_sentences[j]):
                        # Skip the overlapping sentence
                        remaining_sentences = chunk_sentences[1:]
                        if remaining_sentences:
                            merged += " " + " ".join(remaining_sentences)
                        overlap_found = True
                        break
                
                if not overlap_found:
                    merged += " " + chunk
            else:
                merged += " " + chunk
        
        return merged.strip()
    
    def sentences_similar(self, sent1, sent2, threshold=0.7):
        """
        Check if two sentences are similar (for overlap detection)
        """
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity >= threshold
    
    def rule_based_sentence_filtering(self, content):
        """
        Fallback rule-based sentence filtering
        """
        sentences = self.split_into_sentences(content)
        
        filtered_sentences = []
        context_buffer = []  # Buffer to maintain context
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence is economic
            is_economic = self.is_economic_sentence(sentence)
            
            # Check if sentence provides context
            is_context = self.is_context_sentence(sentence)
            
            # Check if sentence is explicitly non-economic
            is_non_economic = self.is_non_economic_sentence(sentence)
            
            if is_non_economic:
                # Skip completely non-economic sentences
                context_buffer = []  # Clear context buffer
                continue
            
            if is_economic:
                # Add buffered context sentences first
                filtered_sentences.extend(context_buffer)
                context_buffer = []
                
                # Add the economic sentence
                filtered_sentences.append(sentence)
            elif is_context:
                # Buffer context sentences
                context_buffer.append(sentence)
                
                # Limit context buffer size
                if len(context_buffer) > 2:
                    context_buffer.pop(0)
            else:
                # Neither economic nor context - check if it's connecting previous economic content
                if filtered_sentences and i < len(sentences) - 1:
                    next_sentence = sentences[i + 1] if i + 1 < len(sentences) else ""
                    if self.is_economic_sentence(next_sentence):
                        # This sentence connects economic content
                        context_buffer.append(sentence)
                    else:
                        # Clear context buffer for non-connecting sentences
                        context_buffer = []
                else:
                    context_buffer = []
        
        # Join filtered sentences
        result = ' '.join(filtered_sentences).strip()
        
        # Post-process to ensure natural flow
        return self.ensure_natural_flow(result)
    
    def split_into_sentences(self, text):
        """
        Split text into sentences
        """
        try:
            return nltk.sent_tokenize(text, language='turkish')
        except:
            # Fallback splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def is_economic_sentence(self, sentence):
        """
        Determine if a sentence contains economic content
        """
        sentence_lower = sentence.lower()
        
        # Count economic terms
        economic_term_count = sum(1 for term in self.core_economic_terms if term in sentence_lower)
        
        # Check for economic patterns
        has_percentage = bool(re.search(r'\d+[,.]?\d*\s*%', sentence))
        has_currency = bool(re.search(r'(tl|lira|dolar|euro|sterlin)', sentence_lower))
        has_numbers_with_units = bool(re.search(r'\d+[,.]?\d*\s*(milyon|milyar|bin|trilyon)', sentence_lower))
        has_economic_verbs = any(verb in sentence_lower for verb in [
            'arttı', 'azaldı', 'yükseldi', 'düştü', 'geriledi', 'büyüdü',
            'küçüldü', 'değişti', 'etkiledi', 'fiyatlandı'
        ])
        
        # Scoring system
        score = 0
        score += economic_term_count * 2
        score += 1 if has_percentage else 0
        score += 1 if has_currency else 0
        score += 1 if has_numbers_with_units else 0
        score += 1 if has_economic_verbs else 0
        
        return score >= 3
    
    def is_context_sentence(self, sentence):
        """
        Determine if a sentence provides important context for economic content
        """
        sentence_lower = sentence.lower()
        
        # Check for context preservation terms
        has_context_terms = any(term in sentence_lower for term in self.context_preservation_terms)
        
        # Check for attribution (who said what)
        has_attribution = bool(re.search(r'(açıkladı|belirtti|söyledi|dedi|ifade etti)', sentence_lower))
        
        # Check for institutional references
        has_institution = any(inst in sentence_lower for inst in [
            'merkez bankası', 'maliye bakanlığı', 'hazine', 'borsa', 'tüik',
            'türkstat', 'imf', 'dünya bankası', 'oecd', 'ab'
        ])
        
        # Check for temporal context
        has_temporal = bool(re.search(r'(bugün|dün|yarın|geçen|önümüzdeki|bu)', sentence_lower))
        
        return has_context_terms or has_attribution or has_institution or (has_temporal and len(sentence.split()) <= 15)
    
    def is_non_economic_sentence(self, sentence):
        """
        Determine if a sentence is explicitly non-economic and should be removed
        """
        sentence_lower = sentence.lower()
        
        # Check for non-economic topics
        non_economic_count = sum(1 for topic in self.non_economic_topics if topic in sentence_lower)
        
        # Check for purely political content without economic implications
        is_pure_politics = any(term in sentence_lower for term in [
            'seçim', 'oy', 'parti', 'muhalefet', 'iktidar', 'siyasi',
            'milletvekili', 'başbakan', 'cumhurbaşkanı'
        ]) and not any(term in sentence_lower for term in self.core_economic_terms[:10])
        
        return non_economic_count >= 2 or is_pure_politics
    
    def ensure_natural_flow(self, text):
        """
        Post-process filtered text to ensure natural flow
        """
        if not text:
            return ""
        
        sentences = self.split_into_sentences(text)
        
        # Remove sentences that start with orphaned conjunctions
        cleaned_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence starts with hanging conjunction
            first_words = sentence.split()[:2]
            if first_words and first_words[0].lower() in ['ve', 'ama', 'fakat', 'ancak'] and i == 0:
                # Remove the conjunction from the beginning
                sentence = ' '.join(sentence.split()[1:])
            
            if sentence and len(sentence.split()) >= 3:
                cleaned_sentences.append(sentence)
        
        result = ' '.join(cleaned_sentences)
        
        # Ensure proper capitalization after periods
        result = re.sub(r'(\. +)([a-z])', lambda m: m.group(1) + m.group(2).upper(), result)
        
        return result.strip()
    
    def validate_filtered_content(self, original, filtered):
        """
        Validate that filtered content maintains quality and relevance
        """
        if not filtered or len(filtered.split()) < 20:
            return False
        
        # Check retention ratio - shouldn't be too aggressive
        retention_ratio = len(filtered.split()) / len(original.split())
        if retention_ratio < 0.2:  # Less than 20% retained
            return False
        
        # Check that economic indicators are still present
        if not self.has_economic_indicators(filtered):
            return False
        
        # Check for sentence fragments (incomplete sentences)
        sentences = self.split_into_sentences(filtered)
        complete_sentences = sum(1 for s in sentences if len(s.split()) >= 5 and s.strip().endswith(('.', '!', '?')))
        
        if complete_sentences < len(sentences) * 0.7:  # Less than 70% complete sentences
            return False
        
        return True