# basic.py — CSCI 570 Fall 2025 Sequence Alignment (Basic DP Version)
#
# Functionality:
#   - Read input_file, output_file from command line
#   - Generate two DNA strings according to project description
#   - Use O(mn) dynamic programming to find optimal sequence alignment (with backtracking)
#   - Measure runtime (milliseconds) and memory usage (KB)
#   - Write results to output file in 5 lines
#
# Notes:
#   - Do not print anything to stdout during program execution
#   - All logic (string generation, algorithm, output) is contained in this file

import sys
import time
import psutil


# ===================== Global Parameters: Gap Penalty & Mismatch Matrix =====================

GAP_PENALTY = 30  # δ = 30

ALPHA = {
    'A': {'A': 0,   'C': 110, 'G': 48,  'T': 94},
    'C': {'A': 110, 'C': 0,   'G': 118, 'T': 48},
    'G': {'A': 48,  'C': 118, 'G': 0,   'T': 110},
    'T': {'A': 94,  'C': 48,  'G': 110, 'T': 0},
}


# ===================== String Generator =====================

def generate_strings_from_file(input_path: str):
    """
    Generate two final strings s_final and t_final from file according to project PDF input format.

    File format:
        s0
        n1
        n2
        ...
        t0
        m1
        m2
        ...

    For each string:
        - Start with base string (s0 or t0)
        - For each integer n read, insert the entire current string after index n
          new_string = S[:n+1] + S + S[n+1:]
    """
    with open(input_path, "r") as f:
        # Remove empty lines & strip whitespace
        lines = [line.strip() for line in f if line.strip() != ""]

    if not lines:
        return "", ""

    idx = 0

    # -------- First string s --------
    s = lines[idx]     # s0
    idx += 1

    insert_positions_s = []
    while idx < len(lines) and lines[idx].isdigit():
        insert_positions_s.append(int(lines[idx]))
        idx += 1

    for pos in insert_positions_s:
        # Insert current string s after index pos
        s = s[:pos + 1] + s + s[pos + 1:]

    # -------- Second string t --------
    # At this point lines[idx] should be t0
    t = lines[idx]     # t0
    idx += 1

    insert_positions_t = []
    while idx < len(lines) and lines[idx].isdigit():
        insert_positions_t.append(int(lines[idx]))
        idx += 1

    for pos in insert_positions_t:
        t = t[:pos + 1] + t + t[pos + 1:]

    return s, t


# ===================== Time & Memory Measurement =====================

def get_process_memory_kb() -> int:
    """
    Return current process memory usage (RSS) in KB.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    return int(mem_info.rss / 1024)


def get_current_time_ms() -> float:
    """
    Return current time in milliseconds.
    """
    return time.perf_counter() * 1000.0


def run_with_measurement(alignment_func, X: str, Y: str, *args, **kwargs):
    """
    Unified wrapper: time measurement + memory measurement + call alignment algorithm.

    alignment_func must return:
        (cost, aligned_X, aligned_Y, before_mem_kb, after_mem_kb)

    Returns:
        (cost, aligned_X, aligned_Y, time_ms, memory_kb_diff)
    """
    start_ms = get_current_time_ms()

    cost, aligned_X, aligned_Y, before_mem_kb, after_mem_kb = alignment_func(X, Y, *args, **kwargs)

    end_ms = get_current_time_ms()

    time_ms = end_ms - start_ms
    mem_diff_kb = float(after_mem_kb - before_mem_kb)

    # In extreme cases may be negative (GC, etc.), take max to prevent negative values
    if mem_diff_kb < 0:
        mem_diff_kb = 0.0

    return cost, aligned_X, aligned_Y, time_ms, mem_diff_kb


# ===================== Basic DP Sequence Alignment Algorithm =====================

def mismatch_cost(a: str, b: str) -> int:
    """
    Return the alignment cost α_{ab} for characters a and b.
    """
    return ALPHA[a][b]


def basic_alignment_dp(X: str, Y: str):
    """
    Standard O(mn) dynamic programming + backtracking to find optimal sequence alignment.

    Definition:
        dp[i][j] = minimum alignment cost for X[0..i-1] and Y[0..j-1]

    Transitions:
        1) match/mismatch:
           dp[i-1][j-1] + mismatch_cost(X[i-1], Y[j-1])
        2) gap in Y:
           dp[i-1][j] + GAP_PENALTY
        3) gap in X:
           dp[i][j-1] + GAP_PENALTY

    Returns:
        (min_cost, aligned_X, aligned_Y)
    """
    before_mem_kb = get_process_memory_kb()

    m = len(X)
    n = len(Y)

    # Build (m+1) x (n+1) DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize: X aligned to empty string / empty string aligned to Y
    for i in range(1, m + 1):
        dp[i][0] = i * GAP_PENALTY
    for j in range(1, n + 1):
        dp[0][j] = j * GAP_PENALTY

    # Fill DP table
    for i in range(1, m + 1):
        xi = X[i - 1]
        for j in range(1, n + 1):
            yj = Y[j - 1]

            cost_match = dp[i - 1][j - 1] + mismatch_cost(xi, yj)
            cost_gap_y = dp[i - 1][j] + GAP_PENALTY  # Insert gap in Y
            cost_gap_x = dp[i][j - 1] + GAP_PENALTY  # Insert gap in X

            dp[i][j] = min(cost_match, cost_gap_y, cost_gap_x)

    min_cost = dp[m][n]

    # Backtrack to construct alignment result
    aligned_X = []
    aligned_Y = []

    i, j = m, n
    while i > 0 or j > 0:
        # Try coming from match/mismatch
        if i > 0 and j > 0:
            xi = X[i - 1]
            yj = Y[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + mismatch_cost(xi, yj):
                aligned_X.append(xi)
                aligned_Y.append(yj)
                i -= 1
                j -= 1
                continue

        # Try coming from inserting gap in Y (X[i-1] ↔ '_')
        if i > 0 and dp[i][j] == dp[i - 1][j] + GAP_PENALTY:
            aligned_X.append(X[i - 1])
            aligned_Y.append('_')
            i -= 1
            continue

        # Try coming from inserting gap in X ('_' ↔ Y[j-1])
        if j > 0 and dp[i][j] == dp[i][j - 1] + GAP_PENALTY:
            aligned_X.append('_')
            aligned_Y.append(Y[j - 1])
            j -= 1
            continue

        # Theoretically should not reach here (one of three cases must match)
        break

    # Backtracking produces reversed sequence, need to reverse
    aligned_X.reverse()
    aligned_Y.reverse()

    after_mem_kb = get_process_memory_kb()

    return min_cost, ''.join(aligned_X), ''.join(aligned_Y), before_mem_kb, after_mem_kb


def main():
    # Must receive two command line arguments: input_file, output_file
    if len(sys.argv) != 3:
        # Project requirement: do not print error messages, exit directly
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 1. Generate two input strings
    X, Y = generate_strings_from_file(input_path)

    # 2. Run basic DP and measure time + memory
    cost, aligned_X, aligned_Y, time_ms, mem_kb = run_with_measurement(
        basic_alignment_dp, X, Y
    )

    # 3. Write output file according to project requirements (strictly 5 lines)
    with open(output_path, "w") as f:
        # 1) Cost of the alignment (Integer)
        f.write(str(int(cost)) + "\n")
        # 2) First string alignment
        f.write(aligned_X + "\n")
        # 3) Second string alignment
        f.write(aligned_Y + "\n")
        # 4) Time in Milliseconds (Float)
        f.write(str(float(time_ms)) + "\n")
        # 5) Memory in Kilobytes (Float)
        f.write(str(float(mem_kb)) + "\n")


if __name__ == "__main__":
    main()
