#!/usr/bin/env python3
"""
Analyze PTO-ISA instruction patterns in test cases and kernels.

This script extracts instruction sequences to identify:
1. Common operation patterns (candidates for fusion)
2. Memory access patterns
3. Reduction-expansion sequences
"""

import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set

PTO_ISA_ROOT = Path(__file__).parent.parent.parent.parent / "code-repos" / "pto-isa"

# PTO instruction pattern
INSTR_PATTERN = re.compile(r'\b(T[A-Z][A-Z_0-9]+)\s*\(')

# Known instruction categories
CATEGORIES = {
    "memory": {"TLOAD", "TSTORE", "TSTORE_FP", "TGATHER", "TGATHERB", "TSCATTER", "MGATHER", "MSCATTER"},
    "layout": {"TMOV", "TMOV_FP", "TTRANS", "TEXTRACT", "TRESHAPE", "TASSIGN"},
    "compute_binary": {"TADD", "TSUB", "TMUL", "TDIV", "TREM", "TAND", "TOR", "TXOR"},
    "compute_scalar": {"TADDS", "TSUBS", "TMULS", "TDIVS", "TREMS", "TANDS", "TORS", "TXORS",
                       "TADDSC", "TSUBSC", "TADDC", "TSUBC"},
    "compute_unary": {"TABS", "TNEG", "TNOT", "TEXP", "TLOG", "TSQRT", "TRSQRT", "TRECIP"},
    "activation": {"TRELU", "TLRELU", "TPRELU"},
    "compare": {"TCMP", "TCMPS", "TSEL", "TSELS", "TMAX", "TMAXS", "TMIN", "TMINS"},
    "reduction": {"TROWSUM", "TROWMAX", "TROWMIN", "TCOLSUM", "TCOLMAX", "TCOLMIN"},
    "expansion": {"TROWEXPAND", "TCOLEXPAND", "TEXPANDS"},
    "fused_expand": {"TROWEXPANDSUB", "TROWEXPANDMUL", "TROWEXPANDDIV"},
    "partition": {"TPARTADD", "TPARTMAX", "TPARTMIN"},
    "matmul": {"TMATMUL", "TMATMUL_ACC", "TMATMUL_BIAS", "TMATMUL_MX"},
    "sort": {"TSORT32", "TMRGSORT"},
    "convert": {"TCVT"},
    "sync": {"TSYNC"},
    "misc": {"TCI", "TFILLPAD", "TPRINT"},
}

def categorize_instr(instr: str) -> str:
    """Get category for an instruction."""
    for cat, instrs in CATEGORIES.items():
        if instr in instrs:
            return cat
    return "unknown"

@dataclass
class InstrSequence:
    """A sequence of instructions found in code."""
    file: str
    function: str
    instructions: List[str]
    categories: List[str]

def extract_instructions(content: str) -> List[str]:
    """Extract PTO instructions from source code."""
    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Find all instruction calls
    matches = INSTR_PATTERN.findall(content)
    return matches

def analyze_file(filepath: Path) -> List[InstrSequence]:
    """Analyze a single file for instruction patterns."""
    sequences = []

    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return sequences

    # Find function definitions
    func_pattern = re.compile(
        r'(?:void|auto|template[^{]*?)\s+(\w+)\s*\([^)]*\)\s*\{',
        re.DOTALL
    )

    # Simple approach: extract all instructions from file
    instrs = extract_instructions(content)
    if instrs:
        cats = [categorize_instr(i) for i in instrs]
        sequences.append(InstrSequence(
            file=str(filepath.relative_to(PTO_ISA_ROOT)),
            function="<file>",
            instructions=instrs,
            categories=cats
        ))

    return sequences

def find_patterns(sequences: List[InstrSequence]) -> Dict[str, int]:
    """Find common instruction patterns (2-gram, 3-gram)."""
    bigrams = Counter()
    trigrams = Counter()

    for seq in sequences:
        instrs = seq.instructions
        for i in range(len(instrs) - 1):
            bigram = f"{instrs[i]} -> {instrs[i+1]}"
            bigrams[bigram] += 1

        for i in range(len(instrs) - 2):
            trigram = f"{instrs[i]} -> {instrs[i+1]} -> {instrs[i+2]}"
            trigrams[trigram] += 1

    return {"bigrams": bigrams, "trigrams": trigrams}

def identify_fusion_opportunities(patterns: Dict) -> List[Dict]:
    """Identify patterns that could benefit from fusion."""
    opportunities = []

    # Look for reduction -> expansion patterns (already have some fused)
    reduction_expansion = [
        ("TROWMAX", "TROWEXPAND", "Consider: already have TROWEXPAND variants"),
        ("TROWSUM", "TROWEXPAND", "Consider: already have TROWEXPAND variants"),
        ("TCOLMAX", "TCOLEXPAND", "Consider: add TCOLEXPAND variants if missing"),
    ]

    # Look for common elementwise chains
    elementwise_chains = [
        ("TSUB", "TEXP", "Potential: TSUBEXP (shift + exp for softmax)"),
        ("TEXP", "TROWSUM", "Potential: TEXPROWSUM (exp + sum for softmax)"),
        ("TMUL", "TADD", "Potential: TMAD (multiply-add)"),
        ("TADD", "TRELU", "Potential: TADDRELU (add + activation)"),
    ]

    bigrams = patterns["bigrams"]

    for b1, b2, note in reduction_expansion + elementwise_chains:
        key = f"{b1} -> {b2}"
        if key in bigrams and bigrams[key] > 0:
            opportunities.append({
                "pattern": key,
                "count": bigrams[key],
                "note": note
            })

    # Look for layout transform overhead
    layout_patterns = [
        ("TMOV", "TMATMUL", "Layout transform before matmul"),
        ("TMOV", "TMOV", "Double layout transform"),
        ("TTRANS", "TMOV", "Transpose + layout transform"),
    ]

    for b1, b2, note in layout_patterns:
        key = f"{b1} -> {b2}"
        if key in bigrams and bigrams[key] > 0:
            opportunities.append({
                "pattern": key,
                "count": bigrams[key],
                "note": note
            })

    return opportunities

def main():
    print("PTO-ISA Instruction Pattern Analysis")
    print("=" * 60)

    # Collect source files
    source_dirs = [
        PTO_ISA_ROOT / "tests" / "cpu" / "st" / "testcase",
        PTO_ISA_ROOT / "kernels",
        PTO_ISA_ROOT / "demos" / "cpu",
    ]

    all_files = []
    for d in source_dirs:
        if d.exists():
            all_files.extend(d.rglob("*.cpp"))

    print(f"\nAnalyzing {len(all_files)} source files...")

    # Analyze all files
    all_sequences = []
    instr_counts = Counter()

    for f in all_files:
        seqs = analyze_file(f)
        all_sequences.extend(seqs)
        for seq in seqs:
            instr_counts.update(seq.instructions)

    print(f"Found {len(all_sequences)} code sections with PTO instructions")
    print(f"Total instruction calls: {sum(instr_counts.values())}")

    # Report instruction usage
    print("\n" + "=" * 60)
    print("INSTRUCTION USAGE (Top 30)")
    print("=" * 60)
    for instr, count in instr_counts.most_common(30):
        cat = categorize_instr(instr)
        print(f"  {instr:20s} {count:5d}  [{cat}]")

    # Find patterns
    patterns = find_patterns(all_sequences)

    print("\n" + "=" * 60)
    print("COMMON INSTRUCTION SEQUENCES (Top 20 Bigrams)")
    print("=" * 60)
    for pattern, count in patterns["bigrams"].most_common(20):
        print(f"  {pattern:40s} {count:5d}")

    print("\n" + "=" * 60)
    print("COMMON TRIGRAMS (Top 15)")
    print("=" * 60)
    for pattern, count in patterns["trigrams"].most_common(15):
        print(f"  {pattern}")
        print(f"    Count: {count}")

    # Identify fusion opportunities
    print("\n" + "=" * 60)
    print("FUSION OPPORTUNITIES")
    print("=" * 60)
    opportunities = identify_fusion_opportunities(patterns)
    for opp in sorted(opportunities, key=lambda x: -x["count"]):
        print(f"\n  Pattern: {opp['pattern']}")
        print(f"  Count: {opp['count']}")
        print(f"  Note: {opp['note']}")

    # Category distribution
    print("\n" + "=" * 60)
    print("INSTRUCTION CATEGORY DISTRIBUTION")
    print("=" * 60)
    cat_counts = Counter()
    for instr, count in instr_counts.items():
        cat = categorize_instr(instr)
        cat_counts[cat] += count

    for cat, count in cat_counts.most_common():
        pct = 100.0 * count / sum(cat_counts.values())
        print(f"  {cat:20s} {count:5d} ({pct:5.1f}%)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
