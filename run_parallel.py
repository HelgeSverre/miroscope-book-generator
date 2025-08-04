import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import lilypad
import yaml
from dotenv import load_dotenv
from lilypad import trace
from mirascope.core import openai, prompt_template
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from logging_config import setup_logging

load_dotenv()
console = Console()

# Configure Lilypad
lilypad.configure(
    project_id=os.environ["LILYPAD_PROJECT_ID"],
    api_key=os.environ["LILYPAD_API_KEY"],
    auto_llm=True,
)

# Configure async OpenAI client with connection pooling and timeouts
limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
timeout = httpx.Timeout(120.0, connect=5.0)
http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
async_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=http_client,
)

# Set up comprehensive logging
progress_logger = setup_logging()
logger = logging.getLogger(__name__)


# Data models
@dataclass
class Chapter:
    number: int
    title: str
    sections: List[str]
    content: Optional[str] = None
    draft_content: Optional[str] = None  # Store initial draft


@dataclass
class BookPart:
    number: int
    title: str
    chapters: List[Chapter]


@dataclass
class Book:
    title: str
    parts: List[BookPart]
    appendices: List[str]
    technologies: List[str]


# Pydantic model for structured outline
class BookOutlineStructure(BaseModel):
    parts: List[dict] = Field(description="List of book parts with chapters")
    appendices: List[str] = Field(description="List of appendices")
    technologies: List[str] = Field(description="List of technologies covered")


# Mirascope prompt functions
@trace(name="generate_outline", versioning="automatic")
@openai.call(
    model="gpt-4o",
    client=async_client,
    call_params={"temperature": 0.7, "timeout": 120},
    json_mode=True,
)
@prompt_template(
    """
    You are a senior technical editor at Packt Publishing creating a book outline.

    Book Title: {book_title}
    Context: {context}

    Create a comprehensive table of contents with:
    - 4-5 main parts grouping related concepts
    - 12-15 chapters total
    - 5-8 subsections per chapter
    - 2-3 appendices
    - List of technologies covered

    Structure:
    - Part I: Foundations (3-4 chapters)
    - Part II-III: Core concepts and advanced techniques
    - Part IV: Real-world projects (2-3 complete applications)
    - Part V: Production concerns

    Return a JSON structure with the following format:
    {{
        "parts": [
            {{
                "number": 1,
                "title": "Part Title",
                "chapters": [
                    {{
                        "number": 1,
                        "title": "Chapter Title",
                        "sections": ["Section 1", "Section 2", ...]
                    }}
                ]
            }}
        ],
        "appendices": ["Appendix A: Title", "Appendix B: Title"],
        "technologies": ["Laravel", "PHP", ...]
    }}
    """
)
async def generate_outline(book_title: str, context: str): ...


@trace(name="generate_initial_chapter_draft", versioning="automatic")
@openai.call(
    model="gpt-4o",
    client=async_client,
    call_params={"temperature": 0.8, "max_tokens": 4000, "timeout": 180},
)
@prompt_template(
    """
    You are writing Chapter {chapter_number}: {chapter_title} for the technical book "{book_title}".

    Book Context: {book_context}

    Chapter Sections:
    {sections}

    Write detailed, engaging, and informative content for this chapter that:
    - Covers all the listed sections thoroughly
    - Includes practical code examples where relevant
    - Uses clear explanations suitable for intermediate developers
    - Includes tips, best practices, and common pitfalls
    - Is approximately 3000-4000 words

    Format in markdown with proper headings for each section.
    """
)
async def generate_initial_chapter_draft(
    book_title: str,
    book_context: str,
    chapter_number: int,
    chapter_title: str,
    sections: str,
): ...


@trace(name="refine_chapter_with_context", versioning="automatic")
@openai.call(
    model="gpt-4o",
    client=async_client,
    call_params={"temperature": 0.5, "max_tokens": 4000, "timeout": 180},
)
@prompt_template(
    """
    Refine Chapter {chapter_number}: {chapter_title} to ensure consistency with surrounding chapters.

    Original Chapter Content: {chapter_content}

    Previous Chapter Summary: {previous_summary}
    Next Chapter Title: {next_chapter_title}

    Book Context: {book_context}

    Refine the content by:
    - Ensuring smooth transitions from the previous chapter
    - Setting up concepts that will be used in the next chapter
    - Maintaining consistent tone and technical level throughout
    - Removing any redundancies with other chapters
    - Adding cross-references to related chapters where appropriate
    - Ensuring code examples follow consistent patterns

    Return the refined chapter content in markdown format.
    """
)
async def refine_chapter_with_context(
    chapter_content: str,
    chapter_number: int,
    chapter_title: str,
    previous_summary: str,
    next_chapter_title: str,
    book_context: str,
): ...


@trace(name="summarize_chapter", versioning="automatic")
@openai.call(
    model="gpt-4o", client=async_client, call_params={"temperature": 0.3, "timeout": 60}
)
@prompt_template(
    """
    Create a brief summary (2-3 paragraphs) of this chapter's key concepts and learnings:

    Chapter Title: {chapter_title}
    Chapter Content: {chapter_content}

    The summary should highlight:
    - Main concepts covered
    - Key skills learned
    - How this prepares readers for upcoming chapters
    """
)
async def summarize_chapter(chapter_title: str, chapter_content: str): ...


class ParallelTechnicalBookGenerator:
    def __init__(self, book_title: str, context: str, max_concurrent: int = 5):
        self.book_title = book_title
        self.context = context
        self.book: Optional[Book] = None
        self.chapter_summaries: Dict[int, str] = {}
        self.max_concurrent = max_concurrent

        # Create output directory
        self.output_dir = Path(
            f"generated_books/{book_title.lower().replace(' ', '_')}_parallel"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trace metadata
        self.trace_metadata = {
            "book_title": book_title,
            "generation_type": "parallel",
            "max_concurrent": max_concurrent,
            "start_time": datetime.now().isoformat(),
        }

    @trace(name="generate_book_outline")
    async def generate_outline(self) -> Book:
        """Step 1: Generate book outline"""
        try:
            with lilypad.span("outline_generation_start") as span:
                span.log("Starting book outline generation")
                span.metadata(self.trace_metadata)

                console.print(
                    Panel.fit(
                        "📋 Phase 1/4: Generating book outline", style="bold blue"
                    )
                )
                progress_logger.start_operation(
                    "outline_generation",
                    f"Generating outline for: {self.book_title}",
                    book_title=self.book_title,
                    category=self.category,
                )

                response = await generate_outline(
                    book_title=self.book_title, context=self.context
                )

                import json

                outline_data = BookOutlineStructure.model_validate(
                    json.loads(response.content)
                )

                # Parse outline into Book structure
                parts = []
                for part_data in outline_data.parts:
                    chapters = []
                    for chapter_data in part_data["chapters"]:
                        chapters.append(
                            Chapter(
                                number=chapter_data["number"],
                                title=chapter_data["title"],
                                sections=chapter_data["sections"],
                            )
                        )
                    parts.append(
                        BookPart(
                            number=part_data["number"],
                            title=part_data["title"],
                            chapters=chapters,
                        )
                    )

                self.book = Book(
                    title=self.book_title,
                    parts=parts,
                    appendices=outline_data.appendices,
                    technologies=outline_data.technologies,
                )

                # Update metadata with outline stats
                total_chapters = sum(len(part.chapters) for part in parts)
                span.metadata(
                    {
                        **self.trace_metadata,
                        "total_parts": len(parts),
                        "total_chapters": total_chapters,
                        "total_appendices": len(outline_data.appendices),
                        "technologies": outline_data.technologies,
                    }
                )

                # Save outline
                self.save_outline()
                console.print("[green]✓ Outline generated and saved![/green]")
                progress_logger.complete_operation(
                    "outline_generation",
                    "Outline generation completed",
                    parts=len(self.book.parts),
                    chapters=sum(len(part.chapters) for part in self.book.parts),
                )

                span.log(
                    f"Outline generated successfully with {total_chapters} chapters across {len(parts)} parts"
                )
                return self.book

        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            with lilypad.span("outline_generation_error") as error_span:
                error_span.log(f"Outline generation failed: {str(e)}")
                error_span.metadata({**self.trace_metadata, "error": str(e)})
            raise

    @trace(name="generate_chapter_initial_draft")
    async def generate_initial_draft(self, chapter: Chapter) -> None:
        """Generate initial draft for a single chapter"""
        try:
            with lilypad.span("chapter_draft_generation") as span:
                chapter_metadata = {
                    **self.trace_metadata,
                    "chapter_number": chapter.number,
                    "chapter_title": chapter.title,
                    "sections_count": len(chapter.sections),
                }
                span.metadata(chapter_metadata)
                span.log(
                    f"Starting draft generation for Chapter {chapter.number}: {chapter.title}"
                )

                sections_text = "\n".join(
                    [f"- {section}" for section in chapter.sections]
                )

                response = await generate_initial_chapter_draft(
                    book_title=self.book_title,
                    book_context=self.context,
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    sections=sections_text,
                )

                chapter.draft_content = response.content

                span.log(
                    f"Draft completed for Chapter {chapter.number}, content length: {len(response.content) if response.content else 0} characters"
                )

                # Save draft immediately - need to find part number
                for part in self.book.parts:
                    if any(ch.number == chapter.number for ch in part.chapters):
                        self.save_chapter_draft(part.number, chapter)
                        break

        except Exception as e:
            logger.error(
                f"Error generating draft for Chapter {chapter.number}: {str(e)}"
            )
            with lilypad.span("chapter_draft_error") as error_span:
                error_span.log(
                    f"Chapter {chapter.number} draft generation failed: {str(e)}"
                )
                error_span.metadata(
                    {
                        **self.trace_metadata,
                        "chapter_number": chapter.number,
                        "chapter_title": chapter.title,
                        "error": str(e),
                    }
                )
            raise

    @trace(name="generate_all_initial_drafts")
    async def generate_all_initial_drafts(self):
        """Step 2: Generate all chapter drafts in parallel"""
        try:
            with lilypad.span("parallel_drafts_generation") as span:
                console.print(
                    Panel.fit(
                        "📝 Phase 2/4: Generating initial chapter drafts in parallel",
                        style="bold blue",
                    )
                )
                progress_logger.start_operation(
                    "draft_generation",
                    "Starting parallel chapter draft generation",
                    total_chapters=sum(len(part.chapters) for part in self.book.parts),
                )

                # Flatten all chapters
                all_chapters = []
                for part in self.book.parts:
                    all_chapters.extend(part.chapters)

                total_chapters = len(all_chapters)

                # Set span metadata
                span.metadata(
                    {
                        **self.trace_metadata,
                        "total_chapters": total_chapters,
                        "max_concurrent": self.max_concurrent,
                        "operation": "parallel_draft_generation",
                    }
                )
                span.log(
                    f"Starting parallel generation of {total_chapters} chapter drafts"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Generating drafts", total=total_chapters)

                    # Use semaphore to limit concurrent calls
                    semaphore = asyncio.Semaphore(self.max_concurrent)
                    completed_chapters = 0

                    async def generate_with_progress(chapter: Chapter):
                        nonlocal completed_chapters
                        async with semaphore:
                            progress.update(
                                task,
                                description=f"Drafting Chapter {chapter.number}: {chapter.title}",
                            )
                            await self.generate_initial_draft(chapter)
                            progress.advance(task)
                            completed_chapters += 1
                            console.print(
                                f"  [green]✓ Chapter {chapter.number} draft completed[/green]"
                            )
                            progress_logger.update_progress(
                                "draft_generation",
                                f"Chapter {chapter.number} draft completed",
                                chapter_number=chapter.number,
                                chapter_title=chapter.title,
                                content_length=len(chapter.draft_content or ""),
                            )

                    # Generate all drafts concurrently
                    await asyncio.gather(
                        *[generate_with_progress(chapter) for chapter in all_chapters]
                    )

                    span.log(
                        f"Completed parallel generation of {completed_chapters}/{total_chapters} chapter drafts"
                    )

        except Exception as e:
            logger.error(f"Error in parallel draft generation: {str(e)}")
            with lilypad.span("parallel_drafts_error") as error_span:
                error_span.log(f"Parallel draft generation failed: {str(e)}")
                error_span.metadata({**self.trace_metadata, "error": str(e)})
            raise

    @trace(name="refine_chapter_with_context")
    async def refine_chapter(
        self, part: BookPart, chapter: Chapter, prev_summary: str, next_title: str
    ) -> str:
        """Refine a chapter with context from surrounding chapters"""
        start_time = datetime.now()
        logger.info(
            f"Starting refinement for Chapter {chapter.number}: {chapter.title}"
        )

        try:
            with lilypad.span("chapter_refinement") as span:
                refinement_metadata = {
                    **self.trace_metadata,
                    "chapter_number": chapter.number,
                    "chapter_title": chapter.title,
                    "part_number": part.number,
                    "part_title": part.title,
                    "has_previous_summary": bool(prev_summary),
                    "next_chapter_title": next_title,
                }
                span.metadata(refinement_metadata)
                span.log(
                    f"Starting refinement for Chapter {chapter.number}: {chapter.title}"
                )

                response = await refine_chapter_with_context(
                    chapter_content=chapter.draft_content,
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    previous_summary=prev_summary,
                    next_chapter_title=next_title,
                    book_context=self.context,
                )

                chapter.content = response.content

                # Generate summary
                summary_response = await summarize_chapter(
                    chapter_title=chapter.title, chapter_content=chapter.content
                )

                refined_content_length = (
                    len(response.content) if response.content else 0
                )
                summary_length = (
                    len(summary_response.content) if summary_response.content else 0
                )

                span.log(
                    f"Chapter {chapter.number} refined: content={refined_content_length} chars, summary={summary_length} chars"
                )

                return summary_response.content

        except Exception as e:
            logger.error(f"Error refining Chapter {chapter.number}: {str(e)}")
            with lilypad.span("chapter_refinement_error") as error_span:
                error_span.log(f"Chapter {chapter.number} refinement failed: {str(e)}")
                error_span.metadata(
                    {
                        **self.trace_metadata,
                        "chapter_number": chapter.number,
                        "chapter_title": chapter.title,
                        "error": str(e),
                    }
                )
            raise

    @trace(name="refine_all_chapters")
    async def refine_all_chapters(self):
        """Step 3: Refine all chapters sequentially with context"""
        try:
            with lilypad.span("sequential_refinement") as span:
                console.print(
                    Panel.fit(
                        "🔧 Phase 3/4: Refining chapters with contextual awareness",
                        style="bold blue",
                    )
                )
                progress_logger.complete_operation(
                    "draft_generation", "All chapter drafts generated"
                )
                progress_logger.start_operation(
                    "refinement", "Starting chapter refinement phase"
                )

                total_chapters = sum(len(part.chapters) for part in self.book.parts)

                span.metadata(
                    {
                        **self.trace_metadata,
                        "total_chapters": total_chapters,
                        "operation": "sequential_chapter_refinement",
                    }
                )
                span.log(f"Starting sequential refinement of {total_chapters} chapters")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Refining chapters", total=total_chapters)
                    refined_count = 0

                    previous_summary = "This is the beginning of the book."

                    for part_idx, part in enumerate(self.book.parts):
                        console.print(
                            f"[cyan]Refining Part {part.number}: {part.title}[/cyan]"
                        )
                        progress_logger.update_progress(
                            "refinement",
                            f"Refining Part {part.number}: {part.title}",
                            part_number=part.number,
                            chapters_in_part=len(part.chapters),
                        )

                        for chapter_idx, chapter in enumerate(part.chapters):
                            # Determine next chapter title
                            next_title = "End of book"
                            if chapter_idx < len(part.chapters) - 1:
                                next_title = part.chapters[chapter_idx + 1].title
                            elif part_idx < len(self.book.parts) - 1:
                                next_part = self.book.parts[part_idx + 1]
                                if next_part.chapters:
                                    next_title = next_part.chapters[0].title

                            desc = f"Refining Chapter {chapter.number}: {chapter.title}"
                            progress.update(task, description=desc)

                            # Refine chapter
                            summary = await self.refine_chapter(
                                part, chapter, previous_summary, next_title
                            )

                            # Save refined chapter
                            self.save_chapter(part, chapter)

                            # Update summary for next chapter
                            self.chapter_summaries[chapter.number] = summary
                            previous_summary = summary

                            refined_count += 1
                            progress.advance(task)
                            console.print(
                                f"  [green]✓ Chapter {chapter.number} refined[/green]"
                            )
                            progress_logger.update_progress(
                                "refinement",
                                f"Chapter {chapter.number} refined",
                                chapter_number=chapter.number,
                                draft_length=len(chapter.draft_content or ""),
                                refined_length=len(chapter.content or ""),
                            )

                    span.log(
                        f"Completed sequential refinement of {refined_count}/{total_chapters} chapters"
                    )

        except Exception as e:
            logger.error(f"Error in sequential refinement: {str(e)}")
            with lilypad.span("sequential_refinement_error") as error_span:
                error_span.log(f"Sequential refinement failed: {str(e)}")
                error_span.metadata({**self.trace_metadata, "error": str(e)})
            raise

    def save_outline(self):
        """Save book outline"""
        outline_path = self.output_dir / "outline.md"

        with open(outline_path, "w") as f:
            f.write(f"# {self.book.title}\n\n")
            f.write("## Table of Contents\n\n")

            for part in self.book.parts:
                f.write(f"### Part {part.number}: {part.title}\n\n")

                for chapter in part.chapters:
                    f.write(f"#### Chapter {chapter.number}: {chapter.title}\n")
                    for section in chapter.sections:
                        f.write(f"- {section}\n")
                    f.write("\n")

            f.write("### Appendices\n\n")
            for appendix in self.book.appendices:
                f.write(f"- {appendix}\n")

            f.write("\n### Technologies Covered\n\n")
            for tech in self.book.technologies:
                f.write(f"- {tech}\n")

    def save_chapter_draft(self, part_number: int, chapter: Chapter):
        """Save chapter draft immediately after generation"""
        chapter_dir = self.output_dir / f"part_{part_number}"
        chapter_dir.mkdir(exist_ok=True)

        draft_path = chapter_dir / f"chapter_{chapter.number:02d}_draft.md"
        with open(draft_path, "w") as f:
            f.write(f"# [DRAFT] Chapter {chapter.number}: {chapter.title}\n\n")
            f.write(chapter.draft_content if chapter.draft_content else "")

        console.print(f"    💾 Saved draft: Chapter {chapter.number}")
        return draft_path

    def save_chapter(self, part: BookPart, chapter: Chapter):
        """Save individual chapter (both draft and final)"""
        chapter_dir = self.output_dir / f"part_{part.number}"
        chapter_dir.mkdir(exist_ok=True)

        # Save draft if not already saved
        if chapter.draft_content:
            draft_path = chapter_dir / f"chapter_{chapter.number:02d}_draft.md"
            if not draft_path.exists():
                with open(draft_path, "w") as f:
                    f.write(f"# [DRAFT] Chapter {chapter.number}: {chapter.title}\n\n")
                    f.write(chapter.draft_content)

        # Save final
        if chapter.content:
            chapter_path = chapter_dir / f"chapter_{chapter.number:02d}.md"
            with open(chapter_path, "w") as f:
                f.write(f"# Chapter {chapter.number}: {chapter.title}\n\n")
                f.write(chapter.content)
            console.print(f"    💾 Saved refined: Chapter {chapter.number}")

    @trace(name="compile_book")
    async def compile_book(self):
        """Step 4: Compile all chapters into a single book"""
        try:
            with lilypad.span("book_compilation") as span:
                console.print(
                    Panel.fit("📖 Phase 4/4: Compiling full book", style="bold blue")
                )
                progress_logger.complete_operation("refinement", "All chapters refined")
                progress_logger.start_operation(
                    "compilation", "Compiling complete book"
                )

                full_book_path = (
                    self.output_dir
                    / f"{self.book.title.lower().replace(' ', '_')}_full.md"
                )

                total_chapters = sum(len(part.chapters) for part in self.book.parts)
                total_content_length = 0

                # Calculate total content length
                for part in self.book.parts:
                    for chapter in part.chapters:
                        if chapter.content:
                            total_content_length += len(chapter.content)

                span.metadata(
                    {
                        **self.trace_metadata,
                        "output_file": str(full_book_path),
                        "total_chapters": total_chapters,
                        "total_content_length": total_content_length,
                        "operation": "book_compilation",
                    }
                )
                span.log(
                    f"Starting book compilation with {total_chapters} chapters, {total_content_length} total characters"
                )

                with open(full_book_path, "w") as f:
                    # Title page
                    f.write(f"# {self.book.title}\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}\n\n")
                    f.write("---\n\n")

                    # Table of contents
                    f.write("# Table of Contents\n\n")
                    for part in self.book.parts:
                        f.write(f"## Part {part.number}: {part.title}\n\n")
                        for chapter in part.chapters:
                            f.write(f"- Chapter {chapter.number}: {chapter.title}\n")
                        f.write("\n")

                    f.write("---\n\n")

                    # Chapters
                    for part in self.book.parts:
                        f.write(f"# Part {part.number}: {part.title}\n\n")

                        for chapter in part.chapters:
                            if chapter.content:
                                f.write(chapter.content)
                            f.write("\n\n---\n\n")

                    # Appendices
                    f.write("# Appendices\n\n")
                    for appendix in self.book.appendices:
                        f.write(f"## {appendix}\n\n")
                        f.write("[Content to be added]\n\n")

                    # Technologies
                    f.write("# Technologies Covered\n\n")
                    for tech in self.book.technologies:
                        f.write(f"- {tech}\n")

                console.print(f"[green]✓ Book compiled to: {full_book_path}[/green]")
                progress_logger.complete_operation(
                    "compilation",
                    "Book compilation completed",
                    output_path=str(full_book_path),
                    total_size=len(full_content),
                )
                span.log(f"Book compilation completed successfully: {full_book_path}")

        except Exception as e:
            logger.error(f"Error compiling book: {str(e)}")
            with lilypad.span("book_compilation_error") as error_span:
                error_span.log(f"Book compilation failed: {str(e)}")
                error_span.metadata({**self.trace_metadata, "error": str(e)})
            raise

    @trace(name="generate_complete_book")
    async def generate(self):
        """Run the complete book generation process"""
        start_time = datetime.now()

        try:
            with lilypad.span("complete_book_generation") as span:
                console.print(
                    Panel.fit(
                        f"🚀 Generating Technical Book (Parallel): {self.book_title}\n"
                        f"Max concurrent generations: {self.max_concurrent}\n"
                        f"Logs directory: logs/",
                        style="bold",
                    )
                )

                # Update metadata with generation start
                generation_metadata = {
                    **self.trace_metadata,
                    "operation": "complete_book_generation",
                    "generation_start": start_time.isoformat(),
                }
                span.metadata(generation_metadata)
                span.log(f"Starting complete book generation for: {self.book_title}")

                # Step 1: Generate outline
                await self.generate_outline()

                # Step 2: Generate all initial drafts in parallel
                await self.generate_all_initial_drafts()

                # Step 3: Refine all chapters with context
                await self.refine_all_chapters()

                # Step 4: Compile book
                await self.compile_book()

                # Calculate final metrics
                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds()
                total_chapters = (
                    sum(len(part.chapters) for part in self.book.parts)
                    if self.book
                    else 0
                )

                # Final metadata update
                final_metadata = {
                    **generation_metadata,
                    "generation_end": end_time.isoformat(),
                    "total_duration_seconds": total_duration,
                    "final_chapters_count": total_chapters,
                    "output_directory": str(self.output_dir),
                }
                span.metadata(final_metadata)

                console.print(
                    Panel.fit(
                        "[bold green]✨ Book generation complete![/bold green]\n"
                        f"📁 Output directory: {self.output_dir}\n"
                        f"⏱️  Total time: {total_duration:.1f} seconds",
                        style="green",
                    )
                )
                progress_logger.log_summary(
                    "Book Generation Complete",
                    {
                        "Title": self.book_title,
                        "Output Directory": str(self.output_dir),
                        "Total Duration": f"{total_duration:.1f}s",
                        "Parts": len(self.book.parts),
                        "Total Chapters": sum(
                            len(part.chapters) for part in self.book.parts
                        ),
                    },
                )

                span.log(
                    f"Book generation completed successfully in {total_duration:.1f} seconds with {total_chapters} chapters"
                )

        except Exception as e:
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            logger.error(f"Error in complete book generation: {str(e)}")

            with lilypad.span("complete_book_generation_error") as error_span:
                error_span.log(
                    f"Complete book generation failed after {total_duration:.1f} seconds: {str(e)}"
                )
                error_span.metadata(
                    {
                        **self.trace_metadata,
                        "error": str(e),
                        "failed_after_seconds": total_duration,
                        "generation_start": start_time.isoformat(),
                        "generation_failed": end_time.isoformat(),
                    }
                )
            raise


def load_books_from_config(config_path: Path = Path("config.yml")) -> List[dict]:
    """Load book configurations from config.yml"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    books = []
    for book in config.get("books", []):
        # Handle null summary
        if book.get("summary") is None:
            book["summary"] = f"A comprehensive guide to {book['title']}"
        books.append(book)

    return books


def display_book_overview(books: List[dict]):
    """Display an overview of books to be generated"""
    table = Table(
        title="📚 Book Generation Overview", show_header=True, header_style="bold cyan"
    )
    table.add_column("№", style="dim", width=3)
    table.add_column("Title", style="white")
    table.add_column("Category", style="yellow")
    table.add_column("Has Summary", style="green")

    for idx, book in enumerate(books, 1):
        has_summary = (
            "✅"
            if book.get("summary")
            and book["summary"] != f"A comprehensive guide to {book['title']}"
            else "❌"
        )
        table.add_row(
            str(idx), book["title"], book.get("category", "unknown"), has_summary
        )

    console.print(table)
    console.print(f"[bold]Total books to generate: {len(books)}[/bold]")


# Usage
@trace(name="main_book_generation")
async def main():
    """Main function to run the book generation process with full tracing"""
    start_time = datetime.now()

    # Load books from config
    books = load_books_from_config()

    # Display overview
    console.print(Panel.fit("📚 Technical Book Generation System", style="bold blue"))
    display_book_overview(books)

    # Track results
    successful_books = []
    failed_books = []
    total_chapters_generated = 0

    try:
        with lilypad.span("batch_book_generation") as span:
            span.metadata(
                {
                    "total_books": len(books),
                    "start_time": start_time.isoformat(),
                }
            )

            # Generate each book
            for idx, book_config in enumerate(books, 1):
                book_start_time = datetime.now()
                book_title = book_config["title"]
                context = book_config["summary"]
                category = book_config.get("category", "unknown")
                max_concurrent = 5

                console.print(
                    Panel.fit(
                        f"[bold cyan]📚 Book {idx} of {len(books)}[/bold cyan]\n"
                        f"[bold]Title:[/bold] {book_title}\n"
                        f"[bold]Category:[/bold] {category}\n"
                        f"[bold]Status:[/bold] 🔄 Starting generation...",
                        border_style="cyan",
                    )
                )
                progress_logger.start_operation(
                    f"book_{idx}",
                    f"Starting book {idx}: {book_title}",
                    book_index=idx,
                    total_books=len(books),
                )

                try:
                    with lilypad.span(f"book_generation_{idx}") as book_span:
                        book_metadata = {
                            "book_index": idx,
                            "book_title": book_title,
                            "category": category,
                            "has_custom_summary": book_config.get("summary")
                            is not None,
                        }
                        book_span.metadata(book_metadata)

                        generator = ParallelTechnicalBookGenerator(
                            book_title=book_title,
                            context=context,
                            max_concurrent=max_concurrent,
                        )

                        await generator.generate()

                        # Track successful generation
                        book_chapters = (
                            sum(len(part.chapters) for part in generator.book.parts)
                            if generator.book
                            else 0
                        )
                        total_chapters_generated += book_chapters

                        book_duration = (
                            datetime.now() - book_start_time
                        ).total_seconds()
                        successful_books.append(
                            {
                                "title": book_title,
                                "chapters": book_chapters,
                                "duration": book_duration,
                                "output_dir": str(generator.output_dir),
                            }
                        )

                        console.print(
                            Panel.fit(
                                f"[bold green]✅ Book {idx} completed successfully![/bold green]\n"
                                f"Chapters: {book_chapters}\n"
                                f"Duration: {book_duration:.1f}s",
                                style="green",
                            )
                        )
                        progress_logger.complete_operation(
                            f"book_{idx}",
                            f"Book {idx} completed: {book_title}",
                            chapters=book_chapters,
                            duration=book_duration,
                        )

                except Exception as e:
                    logger.error(f"Failed to generate book '{book_title}': {str(e)}")
                    failed_books.append(
                        {"title": book_title, "error": str(e), "category": category}
                    )
                    console.print(
                        Panel.fit(
                            f"[bold red]❌ Book {idx} failed: {str(e)}[/bold red]",
                            style="red",
                        )
                    )
                    progress_logger.log_error(
                        f"book_{idx}",
                        e,
                        f"Failed to generate book {idx}: {book_title}",
                        book_title=book_title,
                        category=category,
                    )

            # Display final summary
            total_duration = (datetime.now() - start_time).total_seconds()

            console.print(Panel.fit("📊 Generation Summary", style="bold green"))

            summary_table = Table(show_header=True, header_style="bold")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white")

            summary_table.add_row("Total Books", str(len(books)))
            summary_table.add_row("✅ Successful", str(len(successful_books)))
            summary_table.add_row("❌ Failed", str(len(failed_books)))
            summary_table.add_row("Total Chapters", str(total_chapters_generated))
            summary_table.add_row("Total Duration", f"{total_duration:.1f} seconds")
            summary_table.add_row(
                "Avg per Book",
                f"{total_duration / len(books):.1f} seconds" if books else "N/A",
            )

            console.print(summary_table)

            # List successful books
            if successful_books:
                console.print("[bold green]✅ Successfully Generated:[/bold green]")
                for book in successful_books:
                    console.print(
                        f"   • {book['title']}\n"
                        f"     📁 {book['output_dir']}\n"
                        f"     📊 {book['chapters']} chapters in {book['duration']:.1f}s"
                    )

            # List failed books
            if failed_books:
                console.print("[bold red]❌ Failed Books:[/bold red]")
                for book in failed_books:
                    console.print(
                        f"   • {book['title']} ({book['category']})\n"
                        f"     Error: {book['error']}"
                    )

            progress_logger.log_summary(
                "Generation Session Complete",
                {
                    "Total Books": len(books),
                    "Successful": len(successful_books),
                    "Failed": len(failed_books),
                    "Total Duration": f"{span.duration_seconds:.1f}s"
                    if hasattr(span, "duration_seconds")
                    else "N/A",
                },
            )

            span.metadata(
                {
                    "total_books": len(books),
                    "successful_books": len(successful_books),
                    "failed_books": len(failed_books),
                    "total_chapters": total_chapters_generated,
                    "total_duration_seconds": total_duration,
                }
            )

    except Exception as e:
        logger.error(f"Error in batch book generation: {str(e)}")
        with lilypad.span("batch_generation_error") as error_span:
            error_span.log(f"Batch generation failed: {str(e)}")
            error_span.metadata(
                {"error": str(e), "execution_failed": datetime.now().isoformat()}
            )
        raise


if __name__ == "__main__":
    asyncio.run(main())
