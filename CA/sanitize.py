import nbformat
import re

fname = "GenAI_Practical_2.ipynb"

# characters impossible for LaTeX
BAD = [
    "\uFE0F",               # variation selector (invisible emoji)
    "\u001b",               # ANSI escape prefix
    "│", "┃", "┏", "┓", "┫", "┣", "━", "╇", "╋", "┼", "─", "┤",
]

def clean(s):
    if not isinstance(s, str): 
        return s

    # Remove forbidden Unicode
    for ch in BAD:
        s = s.replace(ch, "")

    # Remove ANSI escape codes: \u001b[32m etc.
    s = re.sub(r"\u001b\[[0-9;]*m", "", s)

    # Replace fancy arrows & symbols
    s = s.replace("→", "\\(\\rightarrow\\)")
    s = s.replace("×", "\\(\\times\\)")

    # Remove literal "\n" inside text
    s = s.replace("\\n", "\n")

    # Remove <span> and HTML
    s = re.sub(r"<[^>]+>", "", s)

    # Remove terminal-style ASCII art lines
    s = re.sub(r"[^A-Za-z0-9 .,!?$()\[\]\n\r*_:=-]+", "", s)

    return s


nb = nbformat.read(fname, as_version=4)

for cell in nb.cells:
    if "source" in cell:
        cell.source = clean(cell.source)
    if "outputs" in cell:
        cell.outputs = []  # remove outputs completely
    cell.execution_count = None

nbformat.write(nb, fname)

print("✔ Notebook cleaned! Now run:")
print("  jupyter nbconvert --to pdf GenAI_Practical_2.ipynb")
