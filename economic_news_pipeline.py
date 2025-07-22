"""
Economic News Pipeline
Complete pipeline for extracting economic content from Turkish news broadcasts
with semantic-aware segmentation and sentence-level filtering
"""

import os
import sys
import json
from datetime import datetime
import traceback

from news_structure_analyzer import NewsStructureAnalyzer
from semantic_segmenter import SemanticSegmenter
from economic_content_filter import EconomicContentFilter

class EconomicNewsPipeline:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.structure_analyzer = NewsStructureAnalyzer()
        self.semantic_segmenter = SemanticSegmenter()
        self.economic_filter = EconomicContentFilter()
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'files_with_economic_content': 0,
            'total_segments_created': 0,
            'total_segments_with_economic_content': 0,
            'total_original_words': 0,
            'total_filtered_words': 0,
            'processing_errors': 0
        }
    
    def process_file(self, file_path):
        """
        Process a single news file through the complete pipeline
        Returns: dict with processing results and statistics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(file_path)}")
            print(f"{'='*60}")
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content or len(content.split()) < 50:
                if self.verbose:
                    print("❌ File too short or empty, skipping")
                return None
            
            original_word_count = len(content.split())
            self.stats['total_original_words'] += original_word_count
            
            if self.verbose:
                print(f"📄 Original content: {original_word_count} words")
            
            # Step 1: Analyze news structure (intro, news, ads, outro)
            if self.verbose:
                print("\n🔍 Step 1: Analyzing news structure...")
            
            structure_sections = self.structure_analyzer.analyze_structure(content)
            
            if not structure_sections:
                if self.verbose:
                    print("❌ Structure analysis failed, treating as single news segment")
                structure_sections = [{
                    'type': 'NEWS',
                    'start_pos': 0,
                    'end_pos': len(content),
                    'description': 'Fallback news section'
                }]
            
            # Extract sections
            extracted_sections = self.structure_analyzer.extract_sections(content, structure_sections)
            
            if self.verbose:
                for section_type, sections in extracted_sections.items():
                    print(f"  📑 {section_type}: {len(sections)} section(s)")
                    for i, section in enumerate(sections):
                        print(f"    Section {i+1}: {len(section['content'].split())} words")
            
            # Step 2: Focus on NEWS sections and segment them semantically
            if self.verbose:
                print("\n🧠 Step 2: Semantic segmentation of news content...")
            
            # Collect all sections as potential news content (except pure intro/outro)
            news_sections = []
            for section_type, sections in extracted_sections.items():
                # Skip only pure intro/outro, include everything else as potential news
                if section_type not in ['INTRO', 'OUTRO'] or len(sections[0]['content'].split()) > 100:
                    news_sections.extend(sections)
            
            # If no sections found, use the whole content as news
            if not news_sections:
                news_sections = [{'content': content, 'start_pos': 0, 'end_pos': len(content), 'description': 'Full content as news'}]
            
            if self.verbose:
                print(f"  📰 Found {len(news_sections)} sections to process as news content")
            if not news_sections:
                if self.verbose:
                    print("❌ No news sections found")
                return None
            
            # Segment news content into coherent stories
            news_segments = self.semantic_segmenter.segment_news_content(news_sections)
            
            if not news_segments:
                if self.verbose:
                    print("❌ Semantic segmentation failed")
                return None
            
            self.stats['total_segments_created'] += len(news_segments)
            
            if self.verbose:
                print(f"  🎯 Created {len(news_segments)} semantic segments")
                for i, segment in enumerate(news_segments):
                    word_count = len(segment['content'].split())
                    method = segment.get('method', 'unknown')
                    confidence = segment.get('confidence', 0)
                    print(f"    Segment {i+1}: {word_count} words (method: {method}, confidence: {confidence:.2f})")
            
            # Step 3: Filter for economic content at sentence level
            if self.verbose:
                print("\n💰 Step 3: Filtering for economic content...")
            
            economic_segments = self.economic_filter.filter_economic_content(news_segments)
            
            if not economic_segments:
                if self.verbose:
                    print("❌ No economic content found in segments")
                return None
            
            self.stats['total_segments_with_economic_content'] += len(economic_segments)
            
            # Calculate statistics
            total_filtered_words = sum(len(seg['content'].split()) for seg in economic_segments)
            self.stats['total_filtered_words'] += total_filtered_words
            
            retention_ratio = total_filtered_words / original_word_count if original_word_count > 0 else 0
            
            if self.verbose:
                print(f"  ✅ Economic segments: {len(economic_segments)}")
                print(f"  📊 Content retention: {retention_ratio:.2%} ({total_filtered_words}/{original_word_count} words)")
                
                for i, segment in enumerate(economic_segments):
                    seg_retention = segment.get('retention_ratio', 0)
                    print(f"    Economic Segment {i+1}: {segment['filtered_length']} words (retention: {seg_retention:.2%})")
            
            # Step 4: Create result structure
            result = {
                'file_path': file_path,
                'processing_timestamp': datetime.now().isoformat(),
                'original_word_count': original_word_count,
                'filtered_word_count': total_filtered_words,
                'retention_ratio': retention_ratio,
                'structure_sections': structure_sections,
                'news_segments_count': len(news_segments),
                'economic_segments_count': len(economic_segments),
                'economic_segments': economic_segments,
                'pipeline_stats': {
                    'structure_analysis_success': len(structure_sections) > 0,
                    'semantic_segmentation_success': len(news_segments) > 0,
                    'economic_filtering_success': len(economic_segments) > 0
                }
            }
            
            # Update global statistics
            self.stats['files_processed'] += 1
            if economic_segments:
                self.stats['files_with_economic_content'] += 1
            
            if self.verbose:
                print(f"\n✅ Processing completed successfully!")
                print(f"📈 Pipeline success: Structure ✓ | Segmentation ✓ | Economic Filter ✓")
            
            return result
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            if self.verbose:
                print(f"\n❌ Processing failed: {str(e)}")
                print(f"🔧 Error details: {traceback.format_exc()}")
            return {
                'file_path': file_path,
                'processing_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'error_traceback': traceback.format_exc()
            }
    
    def save_results(self, file_path, result):
        """
        Save processing results to output files
        """
        if not result or 'error' in result:
            return
        
        os.makedirs("processed_v2", exist_ok=True)
        basename = os.path.basename(file_path).replace('.txt', '')
        
        # Save economic content in readable format
        if result.get('economic_segments'):
            content_file = f"processed_v2/{basename}_economic_v2.txt"
            self.save_economic_content(content_file, result)
            
            if self.verbose:
                print(f"💾 Saved economic content: {content_file}")
        
        # Save detailed metadata
        metadata_file = f"processed_v2/{basename}_metadata.json"
        self.save_metadata(metadata_file, result)
        
        if self.verbose:
            print(f"📋 Saved metadata: {metadata_file}")
    
    def save_economic_content(self, file_path, result):
        """
        Save economic content in readable format
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Economic Content from: {os.path.basename(result['file_path'])}\n")
            f.write(f"# Processed: {result['processing_timestamp']}\n")
            f.write(f"# Original: {result['original_word_count']} words → Filtered: {result['filtered_word_count']} words ({result['retention_ratio']:.2%} retention)\n")
            f.write(f"# Economic segments: {result['economic_segments_count']}\n\n")
            
            # Economic segments
            for i, segment in enumerate(result['economic_segments']):
                f.write(f"## Economic Segment {i+1}\n\n")
                f.write(f"**Method**: {segment.get('method', 'unknown')}\n")
                f.write(f"**Confidence**: {segment.get('confidence', 0):.2f}\n")
                f.write(f"**Word Count**: {segment.get('filtered_length', 'unknown')}\n")
                f.write(f"**Retention**: {segment.get('retention_ratio', 0):.2%}\n\n")
                
                f.write(segment['content'])
                f.write("\n\n" + "="*80 + "\n\n")
    
    def save_metadata(self, file_path, result):
        """
        Save detailed processing metadata as JSON
        """
        # Remove content from segments to reduce file size, keep only metadata
        metadata = result.copy()
        if 'economic_segments' in metadata:
            metadata['economic_segments'] = [
                {
                    'segment_id': seg.get('segment_id'),
                    'method': seg.get('method'),
                    'confidence': seg.get('confidence'),
                    'original_length': seg.get('original_length'),
                    'filtered_length': seg.get('filtered_length'),
                    'retention_ratio': seg.get('retention_ratio'),
                    'section_description': seg.get('section_description')
                }
                for seg in metadata['economic_segments']
            ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def should_process_file(self, file_path, force_reprocess=False):
        """
        Check if file should be processed (not already done or force flag set)
        """
        if force_reprocess:
            return True
        
        basename = os.path.basename(file_path).replace('.txt', '')
        output_file = f"processed_v2/{basename}_economic_v2.txt"
        return not os.path.exists(output_file)
    
    def process_directory(self, directory_path, force_reprocess=False):
        """
        Process all .txt files in a directory
        """
        if not os.path.isdir(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return
        
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        if not files:
            print(f"❌ No .txt files found in {directory_path}")
            return
        
        files_to_process = [f for f in files if self.should_process_file(os.path.join(directory_path, f), force_reprocess)]
        skipped_count = len(files) - len(files_to_process)
        
        print(f"\n📁 Directory: {directory_path}")
        print(f"📄 Files to process: {len(files_to_process)}")
        print(f"⏭️  Already processed: {skipped_count}")
        
        if not files_to_process:
            print("✅ All files already processed. Use --force to reprocess.")
            return
        
        successful_processes = 0
        
        for i, filename in enumerate(files_to_process, 1):
            file_path = os.path.join(directory_path, filename)
            
            if self.verbose:
                print(f"\n📋 Progress: {i}/{len(files_to_process)} files")
            
            result = self.process_file(file_path)
            
            if result and 'error' not in result:
                self.save_results(file_path, result)
                successful_processes += 1
            
            # Show progress
            if not self.verbose and i % 5 == 0:
                success_rate = successful_processes / i * 100
                print(f"Progress: {i}/{len(files_to_process)} files ({success_rate:.1f}% success)")
        
        # Final summary
        success_rate = successful_processes / len(files_to_process) * 100
        print(f"\n📊 Directory processing completed:")
        print(f"   ✅ Successful: {successful_processes}/{len(files_to_process)} ({success_rate:.1f}%)")
        print(f"   ❌ Errors: {len(files_to_process) - successful_processes}")
    
    def process_all_channels(self, news_dir="news", force_reprocess=False):
        """
        Process all news channels automatically
        """
        if not os.path.exists(news_dir):
            print(f"❌ News directory not found: {news_dir}")
            return
        
        # Get all subdirectories in news/ folder
        channels = [d for d in os.listdir(news_dir) 
                   if os.path.isdir(os.path.join(news_dir, d)) and not d.startswith('.')]
        
        if not channels:
            print("❌ No channel directories found in news/ folder")
            return
        
        channels = sorted(channels)
        print(f"📺 Found channels: {', '.join(channels)}")
        
        for channel in channels:
            channel_dir = os.path.join(news_dir, channel)
            print(f"\n🎯 Processing channel: {channel}")
            self.process_directory(channel_dir, force_reprocess)
        
        # Global statistics summary
        self.print_global_statistics()
    
    def print_global_statistics(self):
        """
        Print pipeline statistics
        """
        print(f"\n{'='*60}")
        print(f"📊 PIPELINE STATISTICS")
        print(f"{'='*60}")
        print(f"📄 Files processed: {self.stats['files_processed']}")
        print(f"💰 Files with economic content: {self.stats['files_with_economic_content']}")
        print(f"📈 Success rate: {self.stats['files_with_economic_content']/max(1, self.stats['files_processed']):.2%}")
        print(f"🎯 Total segments created: {self.stats['total_segments_created']}")
        print(f"💼 Economic segments: {self.stats['total_segments_with_economic_content']}")
        print(f"📝 Total words processed: {self.stats['total_original_words']:,}")
        print(f"✨ Economic words extracted: {self.stats['total_filtered_words']:,}")
        print(f"📊 Overall retention rate: {self.stats['total_filtered_words']/max(1, self.stats['total_original_words']):.2%}")
        print(f"❌ Processing errors: {self.stats['processing_errors']}")

def main():
    if len(sys.argv) < 2:
        print("📚 Turkish Economic News Pipeline v2.0")
        print("\nUsage:")
        print(f"  {sys.argv[0]} <file_path>          # Process single file")
        print(f"  {sys.argv[0]} <directory_path>     # Process directory")  
        print(f"  {sys.argv[0]} all                  # Process ALL channels automatically")
        print(f"  {sys.argv[0]} <target> --force     # Force reprocess existing files")
        print(f"  {sys.argv[0]} <target> --quiet     # Minimal output")
        print()
        print("✨ Features:")
        print("  • 🏗️  Semantic-aware news structure analysis")
        print("  • 🧠 Intelligent story segmentation")
        print("  • 💰 Sentence-level economic content filtering")
        print("  • 🔄 Context preservation")
        print("  • 📊 Detailed statistics and metadata")
        print()
        print("📁 Output: processed_v2/ directory")
        return
    
    target = sys.argv[1]
    force_reprocess = "--force" in sys.argv
    quiet_mode = "--quiet" in sys.argv
    
    pipeline = EconomicNewsPipeline(verbose=not quiet_mode)
    
    if target == "all":
        # Process all channels
        pipeline.process_all_channels(force_reprocess=force_reprocess)
    elif os.path.isdir(target):
        # Process directory
        pipeline.process_directory(target, force_reprocess=force_reprocess)
        pipeline.print_global_statistics()
    elif os.path.isfile(target):
        # Process single file
        if pipeline.should_process_file(target, force_reprocess):
            result = pipeline.process_file(target)
            if result:
                pipeline.save_results(target, result)
                
                # Show preview if successful
                if 'economic_segments' in result and result['economic_segments']:
                    print(f"\n📖 Preview of first economic segment:")
                    first_segment = result['economic_segments'][0]
                    preview = first_segment['content'][:300] + "..." if len(first_segment['content']) > 300 else first_segment['content']
                    print(f"Content: {preview}")
        else:
            basename = os.path.basename(target).replace('.txt', '')
            output_file = f"processed_v2/{basename}_economic_v2.txt"
            print(f"✅ File already processed: {output_file}")
            print("Use --force flag to reprocess")
        
        pipeline.print_global_statistics()
    else:
        print(f"❌ Target not found: {target}")

if __name__ == "__main__":
    main()