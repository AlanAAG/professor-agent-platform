#!/usr/bin/env python3
"""
Standalone script to ingest manually-provided transcript files.

This script processes transcript files in the manual directory and embeds them
into the Supabase vector database for RAG retrieval.

Usage: 
    python scripts/ingest_manual.py [--directory PATH] [--dry-run] [--validate-only]

File Format Expected:
    Line 1: {ClassName} - {Lecture Title}
    Line 2: Date: {Date} | Time: {Time}
    Line 3: (empty line)
    Line 4+: Transcript content...

Example:
    Statistics - Lecture 5: Probability Distributions
    Date: 2025-01-15 | Time: 10:00 AM

    0:01 Welcome to the first lecture...
    5:23 Today we'll cover normal distributions...
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.refinery.manual_ingest import ingest_all_manual_transcripts, validate_file_format


def setup_logging(log_file: str = "logs/manual_ingest.log", verbose: bool = False) -> None:
    """
    Setup logging to both console and file.
    
    Args:
        log_file: Path to log file
        verbose: If True, set DEBUG level, otherwise INFO
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set logging level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def validate_directory(directory: str) -> bool:
    """
    Validate that the directory exists and contains .txt files.
    
    Args:
        directory: Path to directory to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logging.error(f"Directory does not exist: {directory}")
        return False
    
    if not dir_path.is_dir():
        logging.error(f"Path is not a directory: {directory}")
        return False
    
    txt_files = list(dir_path.glob("*.txt"))
    if not txt_files:
        logging.warning(f"No .txt files found in {directory}")
        return False
    
    logging.info(f"Found {len(txt_files)} .txt files in {directory}")
    return True


def validate_all_files(directory: str) -> bool:
    """
    Validate format of all .txt files in directory.
    
    Args:
        directory: Directory containing files to validate
        
    Returns:
        bool: True if all files are valid, False otherwise
    """
    dir_path = Path(directory)
    txt_files = list(dir_path.glob("*.txt"))
    
    if not txt_files:
        logging.error("No .txt files found to validate")
        return False
    
    logging.info(f"Validating format of {len(txt_files)} files...")
    
    all_valid = True
    for txt_file in txt_files:
        logging.info(f"\n--- Validating {txt_file.name} ---")
        if not validate_file_format(str(txt_file)):
            all_valid = False
            logging.error(f"❌ Validation failed for {txt_file.name}")
        else:
            logging.info(f"✅ Validation passed for {txt_file.name}")
    
    return all_valid


def print_summary(stats: dict) -> None:
    """
    Print a formatted summary of ingestion results.
    
    Args:
        stats: Statistics dictionary from ingest_all_manual_transcripts
    """
    print("\n" + "="*60)
    print("MANUAL TRANSCRIPT INGESTION SUMMARY")
    print("="*60)
    print(f"Total files processed: {stats['total']}")
    print(f"Successfully embedded: {stats['successful']}")
    print(f"Failed: {len(stats['failed'])}")
    
    if stats['failed']:
        print("\nFailed files:")
        for filename in stats['failed']:
            print(f"  ❌ {filename}")
    
    if stats['successful'] > 0:
        print(f"\n✅ {stats['successful']} transcript(s) successfully embedded into Supabase")
    
    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print("="*60)


def main():
    """Main entry point for the manual ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest manual transcript files into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_manual.py
  python scripts/ingest_manual.py --directory /path/to/transcripts/
  python scripts/ingest_manual.py --dry-run --verbose
  python scripts/ingest_manual.py --validate-only

File Format:
  Each .txt file should have:
  Line 1: {ClassName} - {Lecture Title}
  Line 2: Date: {Date} | Time: {Time}  
  Line 3: (empty line)
  Line 4+: Transcript content...
        """
    )
    
    parser.add_argument(
        "--directory",
        default="data/raw_transcripts/manual/",
        help="Directory containing manual transcript .txt files (default: data/raw_transcripts/manual/)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate files but don't write to database"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate file formats, don't process or embed"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    
    parser.add_argument(
        "--log-file",
        default="logs/manual_ingest.log",
        help="Path to log file (default: logs/manual_ingest.log)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    try:
        setup_logging(args.log_file, args.verbose)
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)
    
    logging.info("="*60)
    logging.info("MANUAL TRANSCRIPT INGESTION STARTED")
    logging.info("="*60)
    logging.info(f"Directory: {args.directory}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info(f"Validate only: {args.validate_only}")
    logging.info(f"Verbose: {args.verbose}")
    
    # Validate directory
    if not validate_directory(args.directory):
        logging.error("Directory validation failed")
        sys.exit(1)
    
    # Validate-only mode
    if args.validate_only:
        logging.info("Running in validate-only mode...")
        if validate_all_files(args.directory):
            print("\n✅ All files passed validation!")
            sys.exit(0)
        else:
            print("\n❌ Some files failed validation. Check logs for details.")
            sys.exit(1)
    
    # Dry-run mode
    if args.dry_run:
        logging.info("Running in dry-run mode...")
        logging.info("Files would be processed but not embedded to database")
        
        # Validate all files first
        if not validate_all_files(args.directory):
            logging.error("File validation failed in dry-run mode")
            sys.exit(1)
        
        print("\n✅ Dry run completed successfully!")
        print("All files passed validation and would be processed.")
        print("Run without --dry-run to actually embed to database.")
        sys.exit(0)
    
    # Normal processing mode
    try:
        logging.info("Starting manual transcript ingestion...")
        stats = ingest_all_manual_transcripts(args.directory)
        
        # Print summary
        print_summary(stats)
        
        # Log final status
        if stats['total'] == 0:
            logging.warning("No files were processed")
            sys.exit(1)
        elif len(stats['failed']) == 0:
            logging.info("All files processed successfully!")
            sys.exit(0)
        elif stats['successful'] > 0:
            logging.warning(f"Partial success: {stats['successful']}/{stats['total']} files processed")
            sys.exit(2)  # Partial failure
        else:
            logging.error("All files failed to process")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        print("\n⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        print("Check logs for full details.")
        sys.exit(1)


if __name__ == "__main__":
    main()