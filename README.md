# Book Generator with Mirascope

An advanced technical book generation system built with Mirascope, featuring asynchronous parallel processing with comprehensive logging and quality assurance.

## Features

### ⚡ Performance Optimizations
- Parallel chapter generation
- Async file operations
- Connection pooling for API calls
- Smart retry mechanisms

## Installation

```bash
# Using uv (recommended)
make install

# Or traditional pip
make pip-install
```

## Configuration

1. Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key
   LILYPAD_PROJECT_ID=your_project_id
   LILYPAD_API_KEY=your_lilypad_key
   ```

2. Define books in `config.yml`:
   ```yaml
   books:
     - title: "Your Book Title"
       summary: "Book context and description"
       category: "framework_specific"
   ```

## Usage

```bash
# Run the generator
make run
# Or directly
python run_parallel.py
```

## Output Structure

```
generated_books/
├── book_name/
│   ├── outline.md              # Table of contents
│   ├── part_N/                 # Chapter drafts and final versions
│   └── book_name_full.md       # Complete compiled book
```

## Development

```bash
make check   # Check code quality
make format  # Format code
make clean   # Remove generated books
```

## Key Features

- **Parallel Processing**: Generate multiple chapters simultaneously
- **Smart Retry Logic**: Robust error handling with exponential backoff  
- **Comprehensive Logging**: Detailed file and console logging with progress tracking
- **Connection Pooling**: Optimized HTTP client configuration
- **Context-Aware Refinement**: Chapters refined with awareness of surrounding content

## License

MIT