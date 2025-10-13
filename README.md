# Scan-to-Booklet

Convert scanned saddle-stitched booklet PDFs into properly ordered pages for duplex printing.

## Problem

When you scan a saddle-stitched booklet without removing the staples, each scanned page contains two booklet pages side-by-side. The pages are also in the wrong order for sequential reading or printing.

## Solution

This script:
1. Splits each scanned page vertically into left and right halves
2. Reorders the pages so they can be printed as a proper booklet

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python booklet_reorder.py input.pdf output.pdf
```

### Example

```bash
python booklet_reorder.py scanned_manual.pdf reordered_manual.pdf
```

## How It Works

### Input Format
For a saddle-stitched booklet scanned sequentially without removing staples:
- **Scan 1**: back cover (left) | front cover (right)
- **Scan 2**: page 2 (left) | page 3 (right)
- **Scan 3**: page 4 (left) | page 5 (right)
- And so on...

### Processing Steps
The script:
1. Splits each scanned page vertically into left and right halves
2. Extracts pages in reading order (front cover, page 2, page 3, ..., back cover)
3. Arranges pages for duplex booklet printing with 2 booklet pages per PDF page
4. Pads with blank pages if needed (total pages must be divisible by 4)

### Output Format
Each output PDF page contains 2 booklet pages side-by-side, arranged so that when printed duplex and folded, the pages are in correct order.

For example, with 8 pages:
- **Sheet 1 front**: page 8 | page 1
- **Sheet 1 back**: page 2 | page 7
- **Sheet 2 front**: page 6 | page 3
- **Sheet 2 back**: page 4 | page 5

## Printing the Result

After running the script:
1. Open the output PDF in your PDF viewer
2. Print using **duplex/double-sided printing**
3. Set to flip on **SHORT edge** (also called "tumble" or "short-edge binding")
4. Print all pages in order
5. Stack the printed sheets in the order they came out
6. Fold all sheets in half together down the middle
7. Staple along the center fold

The result will be a properly ordered booklet!

## Requirements

- Python 3.7+
- PyPDF2
- PyMuPDF (fitz)
- Pillow

## License

MIT
