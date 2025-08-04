# Book Generator with Mirascope

An asynchronous Python tool that generates technical books using OpenAI's GPT-4 with parallel processing and
context-aware refinement.

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

## Output

Generated books are saved in `generated_books/` with:

- `outline.md` - Table of contents
- `part_N/` - Chapter drafts and final versions
- `*_full.md` - Complete compiled book

## Development

```bash
make check   # Check code quality
make format  # Format code
make clean   # Remove generated books
```