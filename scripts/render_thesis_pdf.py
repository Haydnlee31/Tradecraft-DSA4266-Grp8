"""Render the thesis review copy; optional ReportLab dependency, no model imports.

This intentionally supports only the small Markdown subset used by chapter.md.
The Markdown and SVG figures remain the editable thesis sources. PDF generation
does not modify training evidence or install anything into the training runtime.
"""
import argparse
import html
from pathlib import Path
import re


def inline(text):
    """Escape prose, preserving only explicitly supported links and emphasis."""
    text = html.escape(text, quote=True)
    def link(match):
        label, target = match.groups()
        if target.startswith(('https://', 'http://')):
            return f'<link href="{target}" color="#0072B2">{label}</link>'
        return label  # Relative source links remain usable in the Markdown copy.
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text


def render(source, output):
    # Keep document-only packages optional, and independent of training imports.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
        Image, PageBreak, KeepTogether, Table, TableStyle)

    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Preserve existing PDF; choose a new output path')
    width = A4[0] - 36 * mm
    body = ParagraphStyle('Body', fontName='Helvetica', fontSize=10.2,
        leading=14.5, textColor=colors.HexColor('#203344'), spaceAfter=9)
    caption = ParagraphStyle('Caption', parent=body, fontSize=8.6, leading=11.5,
        textColor=colors.HexColor('#566875'), spaceAfter=12)
    table_body = ParagraphStyle('Table body', parent=body, fontSize=9,
        leading=12, spaceAfter=0)
    table_head = ParagraphStyle('Table header', parent=table_body,
        fontName='Helvetica-Bold', textColor=colors.white)
    levels = {
        1: ParagraphStyle('Title', parent=body, fontName='Helvetica-Bold',
            fontSize=23, leading=28, spaceAfter=16, keepWithNext=True),
        2: ParagraphStyle('Section', parent=body, fontName='Helvetica-Bold',
            fontSize=16, leading=20, spaceBefore=5, spaceAfter=12, keepWithNext=True),
        3: ParagraphStyle('Subsection', parent=body, fontName='Helvetica-Bold',
            fontSize=11, leading=15, spaceBefore=8, spaceAfter=7, keepWithNext=True),
    }
    lines = source.read_text(encoding='utf-8').splitlines()
    story = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); i += 1; continue
        heading = re.fullmatch(r'(#{1,3}) (.+)', line)
        if heading:
            story.append(Paragraph(inline(heading[2]), levels[len(heading[1])]))
            i += 1; continue
        picture = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if picture:
            path = source.parent / picture[2]
            graphic = Image(str(path))
            ratio = min(width / graphic.imageWidth, 167 * mm / graphic.imageHeight)
            graphic.drawWidth = graphic.imageWidth * ratio
            graphic.drawHeight = graphic.imageHeight * ratio
            graphic.hAlign = 'CENTER'
            group = [graphic, Spacer(1, 5)]
            i += 1
            while i < len(lines) and not lines[i].strip(): i += 1
            if i < len(lines) and re.match(r'Figure \d+\.', lines[i]):
                group.append(Paragraph(inline(lines[i]), caption)); i += 1
            story.append(KeepTogether(group)); continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells): rows.append(cells)
                i += 1
            if not rows or any(len(r) != len(rows[0]) for r in rows):
                raise ValueError('Malformed Markdown table')
            widths = [width * .43] + [width * .57 / (len(rows[0])-1)] * (len(rows[0])-1)
            table = Table([[Paragraph(inline(c), table_head if k == 0 else table_body)
                for c in row] for k, row in enumerate(rows)], colWidths=widths, repeatRows=1,
                hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#203344')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#EEF3F7'), colors.white]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 7),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ]))
            story.extend([table, Spacer(1, 12)]); continue
        # Individual list entries are retained as paragraphs, including numbering.
        paragraph = [line]; i += 1
        if not re.match(r'\d+\. ', line):
            while i < len(lines) and lines[i].strip() and not re.match(r'[#|!]|<!--|\d+\. ', lines[i]):
                paragraph.append(lines[i].strip()); i += 1
        story.append(Paragraph(inline(' '.join(paragraph)), body))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#CDD8DF'))
        canvas.line(18 * mm, 14 * mm, A4[0]-18 * mm, 14 * mm)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#566875'))
        canvas.drawString(18 * mm, 9.5 * mm, 'TRADECRAFT  |  Methods, results and discussion draft')
        canvas.drawRightString(A4[0]-18 * mm, 9.5 * mm, str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=17*mm, bottomMargin=20*mm,
        title='Tradecraft: shared optimization for federated intrusion classification',
        author='Tradecraft project', subject='Illustrated methods, results and discussion draft')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(render(args.source, args.output))
