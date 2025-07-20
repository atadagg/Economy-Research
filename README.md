# Turkish Economic News Processor

An intelligent NLP system that extracts economic content from Turkish news transcriptions using AI-powered topic segmentation and keyword filtering.

## Features

- **Smart Topic Segmentation**: Uses OpenAI GPT-4o-mini to distinguish between different news topics vs. guest changes within the same topic
- **Economic Content Detection**: Combines Turkish economic keywords with AI classification for high-precision extraction  
- **Resume Functionality**: Automatically skips already processed files to save time and API costs
- **Multi-Channel Support**: Works with A Haber, ATV, Halk TV, and other Turkish news sources
- **Flow Preservation**: Maintains natural discourse flow by keeping complete economic discussions together

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key:**
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Commands

#### Process a Single File
```bash
python3.11 news_processor.py "news/AHaber/20241224_Asgari ücrete yapılan zam.tr.txt"
```

#### Process All Files in a Directory  
```bash
python3.11 news_processor.py news/AHaber/
```

#### Process All News Files (All Channels)
```bash
python3.11 news_processor.py all
```

#### Force Reprocess Already-Processed Files
```bash
python3.11 news_processor.py news/AHaber/ --force
```

### Resume Functionality

The system automatically **skips files that have already been processed**:

```bash
# First run - processes all files
python3.11 news_processor.py news/AHaber/
# Output: AHaber: 45 to process, 0 already done

# Second run - skips processed files  
python3.11 news_processor.py news/AHaber/
# Output: AHaber: 0 to process, 45 already done

# Force reprocessing everything
python3.11 news_processor.py news/AHaber/ --force
# Output: AHaber: 45 to process, 0 already done
```

**Benefits:**
- ⏰ **Saves Time**: No duplicate processing
- 💰 **Saves Money**: No duplicate OpenAI API calls  
- 🔄 **Resumable**: Interrupt anytime and continue later

### Output Structure

Processed files are saved in the `processed/` directory:

```
processed/
├── 20241224_Asgari ücrete yapılan zam_economic.txt
├── 20250114_ABD 2025'e kaoslarla başladı_economic.txt
└── ...
```

Each output file contains:
- **Header**: Source file info and segment count
- **Segments**: Economic content with confidence scores  
- **Quality Metadata**: AI confidence levels for each segment

Example output format:
```
# Economic content from: 20241224_Asgari ücrete yapılan zam.tr.txt
# Found 1 economic segments

## Segment 1 (Confidence: 0.95)

asgari ücret artık belirlendi 2025 yılında uygulanacak asgari ücret 22.04 lira...

==================================================
```

## How It Works

### 1. Topic Segmentation
- **AI Analysis**: GPT-4o-mini determines if content discusses one topic or multiple topics
- **Smart Splitting**: Keeps guest interviews about the same topic together
- **Natural Boundaries**: Only splits on genuine topic transitions

### 2. Economic Content Detection  
- **Keyword Filtering**: Pre-filters using Turkish economic terms (enflasyon, asgari ücret, etc.)
- **AI Classification**: GPT-4o-mini classifies relevance with confidence scores
- **Quality Threshold**: Only content above 70% confidence is included

### 3. Quality Control
- **High Precision**: Combines multiple filters to avoid false positives
- **Confidence Scoring**: Every segment includes AI confidence level
- **Flow Preservation**: Complete economic discussions remain intact

## Configuration

Key settings in `config.py`:

```python
# Segmentation parameters
SEGMENT_MIN_LENGTH = 50      # Minimum words per segment
SEGMENT_MAX_LENGTH = 300     # Maximum words per segment  
ECONOMIC_THRESHOLD = 0.7     # Minimum confidence for economic content

# Turkish economic keywords
ECONOMIC_KEYWORDS = [
    "ekonomi", "enflasyon", "asgari ücret", "döviz", "borsa",
    "faiz", "merkez bankası", "büyüme", "ihracat", "ithalat"
    # ... full list of 50+ terms
]
```

## File Structure

```
ekonomi/
├── config.py              # Configuration and keywords
├── news_processor.py      # Main processor script  
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── news/                 # Input news transcriptions
│   ├── AHaber/
│   ├── ATV/ 
│   └── Halk/
└── processed/            # Generated economic content
```

## Example Workflows

### Research Dataset Creation
```bash
# Process entire news archive
python3.11 news_processor.py all

# Check results
find processed -name "*economic.txt" | wc -l
wc -w processed/*economic.txt
```

### Incremental Processing
```bash
# Add new files to news/AHaber/
# Run processor - only new files will be processed
python3.11 news_processor.py news/AHaber/
```

### Quality Validation
```bash
# Process single file and review output
python3.11 news_processor.py "news/AHaber/sample_file.tr.txt"

# Check confidence scores in output file
head -20 processed/sample_file_economic.txt
```

## Performance Notes

- **API Costs**: ~$0.001-0.003 per file (depending on length)
- **Processing Speed**: ~5-10 seconds per file  
- **Resume Feature**: Drastically reduces costs for large datasets
- **Memory Usage**: Minimal - processes one file at a time

## Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**"OpenAI API key not found":**
Check your `.env` file contains:
```
OPENAI_API_KEY=your_key_here
```

**No output files generated:**
- Check if input files contain economic keywords
- Review AI topic analysis output in console
- Files without sufficient economic content won't generate outputs

**Want to reprocess existing files:**
Use the `--force` flag:
```bash
python3.11 news_processor.py news/AHaber/ --force
```

## Output Quality

Expected results:
- **High-confidence economic content** (0.85-0.95 confidence)
- **Complete discussions** kept together as single segments  
- **Clean topic boundaries** when multiple topics exist
- **Natural language flow** preserved for NLP analysis

The system is designed for Turkish economic NLP research requiring high-quality, coherent text segments. 