#!/usr/bin/env python3
"""
Reorder scanned saddle-stitched booklet pages for duplex printing.

This script takes a PDF scan of a saddle-stitched booklet (scanned without removing staples)
and splits each page into left/right halves, then reorders them for proper duplex printing.

For a saddle-stitched booklet scanned sequentially:
- Scan page 1 contains: back cover (left) | front cover (right)
- Scan page 2 contains: page 2 (left) | page 3 (right)
- Scan page 3 contains: page 4 (left) | page 5 (right)
- etc.
"""

import argparse
import sys
from pathlib import Path

try:
    import PyPDF2
    from PIL import Image
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"Error: Missing required library. Please install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def split_page_vertical(page, pdf_writer):
    """
    Split a PDF page vertically into left and right halves.

    Args:
        page: PyPDF2 page object
        pdf_writer: PyPDF2 PdfWriter object to add pages to

    Returns:
        tuple: (left_page, right_page)
    """
    # Get page dimensions
    mediabox = page.mediabox
    width = float(mediabox.width)
    height = float(mediabox.height)

    # Create left half
    left_page = pdf_writer.add_blank_page(width=width/2, height=height)
    left_page.merge_page(page)
    left_page.mediabox.upper_right = (width/2, height)

    # Create right half
    right_page = pdf_writer.add_blank_page(width=width/2, height=height)
    right_page.merge_page(page)
    right_page.mediabox.upper_left = (width/2, height)

    return left_page, right_page


def reorder_booklet_pages(input_pdf_path, output_pdf_path):
    """
    Process a scanned saddle-stitched booklet and reorder pages for duplex printing.

    Args:
        input_pdf_path: Path to input PDF
        output_pdf_path: Path to output PDF
    """
    print(f"Reading PDF: {input_pdf_path}")

    # Use PyMuPDF for better page manipulation
    doc = fitz.open(input_pdf_path)
    num_scanned_pages = len(doc)

    print(f"Found {num_scanned_pages} scanned pages")
    print(f"This will create {num_scanned_pages * 2} booklet pages")

    # Split each scanned page into left and right halves
    all_split_pages = []

    for i in range(num_scanned_pages):
        print(f"Splitting scanned page {i+1}/{num_scanned_pages}...", end='\r')

        page = doc[i]
        rect = page.rect
        width = rect.width
        height = rect.height

        # Create two new pages for left and right halves
        # Left half
        left_doc = fitz.open()
        left_page = left_doc.new_page(width=width/2, height=height)
        left_page.show_pdf_page(left_page.rect, doc, i, clip=fitz.Rect(0, 0, width/2, height))

        # Right half
        right_doc = fitz.open()
        right_page = right_doc.new_page(width=width/2, height=height)
        right_page.show_pdf_page(right_page.rect, doc, i, clip=fitz.Rect(width/2, 0, width, height))

        all_split_pages.append((left_doc, right_doc))

    print(f"\nSplit all pages successfully")

    # Extract pages in reading order first
    # Scan 1: back cover (left) | front cover (right)
    # Scan 2: page 2 (left) | page 3 (right)
    # Scan 3: page 4 (left) | page 5 (right)
    # etc.

    sequential_pages = []
    # Front cover (right side of scan 1)
    sequential_pages.append(all_split_pages[0][1])

    # Pages 2, 3, 4, 5, etc.
    for i in range(1, num_scanned_pages):
        left_doc, right_doc = all_split_pages[i]
        sequential_pages.append(left_doc)   # page 2, 4, 6, etc.
        sequential_pages.append(right_doc)  # page 3, 5, 7, etc.

    # Back cover (left side of scan 1)
    sequential_pages.append(all_split_pages[0][0])

    total_pages = len(sequential_pages)
    print(f"Total pages in reading order: {total_pages}")

    # Get dimensions from first page
    first_page = sequential_pages[0][0]
    page_width = first_page.rect.width
    page_height = first_page.rect.height

    # Pad to multiple of 4 if needed (for booklet printing)
    while len(sequential_pages) % 4 != 0:
        # Add blank pages
        blank_doc = fitz.open()
        blank_doc.new_page(width=page_width, height=page_height)
        sequential_pages.append(blank_doc)
        print(f"Added blank page (now {len(sequential_pages)} pages)")

    print("Arranging pages for duplex booklet printing...")

    # Arrange pages for duplex booklet printing
    # For booklet: each sheet has 4 pages (2 on front, 2 on back)
    # Sheet layout (for n pages):
    #   Front of sheet 1: [n, 1]
    #   Back of sheet 1:  [2, n-1]
    #   Front of sheet 2: [n-2, 3]
    #   Back of sheet 2:  [4, n-3]
    #   etc.
    #
    # Since printer outputs page 1 face-up, we need to reverse the order
    # so the last page printed (covers) ends up on top face-up.

    temp_doc = fitz.open()
    n = len(sequential_pages)
    num_sheets = n // 4

    for sheet in range(num_sheets):
        # Calculate page indices (0-based)
        front_right = sheet * 2                    # pages 0, 2, 4, ...
        front_left = n - 1 - (sheet * 2)           # pages n-1, n-3, n-5, ...
        back_left = sheet * 2 + 1                  # pages 1, 3, 5, ...
        back_right = n - 2 - (sheet * 2)           # pages n-2, n-4, n-6, ...

        print(f"Sheet {sheet+1}: Front[{front_left+1}, {front_right+1}] Back[{back_left+1}, {back_right+1}]")

        # Create front page with two booklet pages side by side
        front_page = temp_doc.new_page(width=page_width * 2, height=page_height)
        # Add left half (higher page number)
        front_page.show_pdf_page(
            fitz.Rect(0, 0, page_width, page_height),
            sequential_pages[front_left],
            0
        )
        # Add right half (lower page number)
        front_page.show_pdf_page(
            fitz.Rect(page_width, 0, page_width * 2, page_height),
            sequential_pages[front_right],
            0
        )

        # Create back page with two booklet pages side by side
        back_page = temp_doc.new_page(width=page_width * 2, height=page_height)
        # Add left half (lower page number)
        back_page.show_pdf_page(
            fitz.Rect(0, 0, page_width, page_height),
            sequential_pages[back_left],
            0
        )
        # Add right half (higher page number)
        back_page.show_pdf_page(
            fitz.Rect(page_width, 0, page_width * 2, page_height),
            sequential_pages[back_right],
            0
        )

    # No reversal needed - printer outputs first page on top,
    # so covers at the beginning will end up on top
    print(f"Writing output PDF: {output_pdf_path}")
    temp_doc.save(output_pdf_path)
    temp_doc.close()

    # Close all temporary documents
    doc.close()
    for left_doc, right_doc in all_split_pages:
        left_doc.close()
        right_doc.close()

    print(f"✓ Successfully created {output_pdf_path}")
    print(f"  Output pages: {num_sheets * 2} (each page contains 2 booklet pages)")
    print(f"  This will print on {num_sheets} sheets of paper")
    print(f"\nTo print as booklet:")
    print(f"  1. Print the PDF using duplex/double-sided printing")
    print(f"  2. Flip on SHORT edge (tumble)")
    print(f"  3. Let the sheets stack as they print (don't reorder)")
    print(f"  4. The covers will be on top, face-up")
    print(f"  5. Fold all sheets in half together")
    print(f"  6. Staple along the center fold")


def main():
    parser = argparse.ArgumentParser(
        description='Reorder scanned saddle-stitched booklet pages for duplex printing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  %(prog)s input.pdf output.pdf
  %(prog)s scanned_manual.pdf reordered_manual.pdf

The script assumes the booklet was scanned sequentially without removing staples:
  Scan 1: back cover | front cover
  Scan 2: page 2 | page 3
  Scan 3: page 4 | page 5
  etc.
        '''
    )

    parser.add_argument('input', help='Input PDF file (scanned booklet)')
    parser.add_argument('output', help='Output PDF file (reordered for printing)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not input_path.suffix.lower() == '.pdf':
        print(f"Error: Input file must be a PDF")
        sys.exit(1)

    # Process the booklet
    try:
        reorder_booklet_pages(args.input, args.output)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
