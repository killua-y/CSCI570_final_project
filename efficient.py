# efficient.py — CSCI 570 Fall 2025 Sequence Alignment (Memory Efficient / Divide & Conquer)
#
# Functionality:
#   - Read input_file, output_file from command line
#   - Generate two DNA strings s and t
#   - Use Divide and Conquer (Hirschberg's Algorithm) to align sequences
#     using O(min(m,n)) space complexity.
#   - Measure runtime and memory usage.
#   - Write results to output file.

import sys
import time
import psutil

# ===================== Global Parameters =====================

GAP_PENALTY = 30

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
    """
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    if not lines:
        return "", ""

    idx = 0

    # -------- First string s --------
    s = lines[idx]
    idx += 1

    insert_positions_s = []
    while idx < len(lines) and lines[idx].isdigit():
        insert_positions_s.append(int(lines[idx]))
        idx += 1

    for pos in insert_positions_s:
        s = s[:pos + 1] + s + s[pos + 1:]

    # -------- Second string t --------
    t = lines[idx]
    idx += 1

    insert_positions_t = []
    while idx < len(lines) and lines[idx].isdigit():
        insert_positions_t.append(int(lines[idx]))
        idx += 1

    for pos in insert_positions_t:
        t = t[:pos + 1] + t + t[pos + 1:]

    return s, t

# ===================== Measurements =====================

def get_process_memory_kb() -> int:
    process = psutil.Process()
    mem_info = process.memory_info()
    return int(mem_info.rss / 1024)

def get_current_time_ms() -> float:
    return time.time() * 1000.0

def run_with_measurement(alignment_func, X: str, Y: str):
    before_mem_kb = get_process_memory_kb()
    start_ms = get_current_time_ms()

    cost, aligned_X, aligned_Y = alignment_func(X, Y)

    end_ms = get_current_time_ms()
    after_mem_kb = get_process_memory_kb()

    time_ms = end_ms - start_ms
    mem_diff_kb = float(after_mem_kb - before_mem_kb)
    
    # Logic note: Python's GC can sometimes cause negative diffs if objects are collected
    # during execution. For this assignment context, we clamp to 0.
    if mem_diff_kb < 0:
        mem_diff_kb = 0.0

    return cost, aligned_X, aligned_Y, time_ms, mem_diff_kb

# ===================== Helper: Basic DP (Base Case) =====================

def basic_alignment_dp(X: str, Y: str):
    """
    Standard O(mn) DP. Used as the base case for Divide & Conquer
    when the problem size is small (len(X) <= 2).
    """
    m = len(X)
    n = len(Y)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * GAP_PENALTY
    for j in range(1, n + 1):
        dp[0][j] = j * GAP_PENALTY

    for i in range(1, m + 1):
        xi = X[i - 1]
        for j in range(1, n + 1):
            yj = Y[j - 1]
            cost_match = dp[i - 1][j - 1] + ALPHA[xi][yj]
            cost_gap_y = dp[i - 1][j] + GAP_PENALTY
            cost_gap_x = dp[i][j - 1] + GAP_PENALTY
            dp[i][j] = min(cost_match, cost_gap_y, cost_gap_x)

    min_cost = dp[m][n]

    aligned_X = []
    aligned_Y = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if dp[i][j] == dp[i - 1][j - 1] + ALPHA[X[i - 1]][Y[j - 1]]:
                aligned_X.append(X[i - 1])
                aligned_Y.append(Y[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + GAP_PENALTY:
            aligned_X.append(X[i - 1])
            aligned_Y.append('_')
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + GAP_PENALTY:
            aligned_X.append('_')
            aligned_Y.append(Y[j - 1])
            j -= 1
            continue

    return min_cost, "".join(aligned_X[::-1]), "".join(aligned_Y[::-1])

# ===================== Memory Efficient Logic =====================

def get_dp_row(X: str, Y: str):
    """
    Computes only the LAST row of the DP matrix for aligning X and Y.
    Space Complexity: O(len(Y))
    Time Complexity: O(len(X) * len(Y))
    """
    n = len(Y)
    # prev_row represents dp[i-1][0...n]
    prev_row = [j * GAP_PENALTY for j in range(n + 1)]
    # curr_row represents dp[i][0...n]
    curr_row = [0] * (n + 1)

    for i in range(1, len(X) + 1):
        curr_row[0] = i * GAP_PENALTY
        xi = X[i-1]
        for j in range(1, n + 1):
            yj = Y[j-1]
            cost_match = prev_row[j - 1] + ALPHA[xi][yj]
            cost_gap_y = prev_row[j] + GAP_PENALTY
            cost_gap_x = curr_row[j - 1] + GAP_PENALTY
            
            curr_row[j] = min(cost_match, cost_gap_y, cost_gap_x)
        
        # Move current row to previous for next iteration
        prev_row[:] = curr_row[:]

    return prev_row

def divide_and_conquer_alignment(X: str, Y: str):
    """
    Hirschberg's Algorithm for Memory Efficient Global Alignment.
    Returns: (cost, aligned_X, aligned_Y)
    """
    m = len(X)
    n = len(Y)

    # Base case: if X is very small, standard DP is sufficient and simpler
    if m <= 2 or n == 0:
        return basic_alignment_dp(X, Y)

    # Divide
    mid_x = m // 2

    # Conquer (Cost Calculation Steps)
    # 1. Calculate score of aligning first half of X with all of Y
    score_left = get_dp_row(X[:mid_x], Y)

    # 2. Calculate score of aligning second half of X (reversed) with all of Y (reversed)
    score_right = get_dp_row(X[mid_x:][::-1], Y[::-1])

    # 3. Find the optimal split point k in Y
    # The optimal path passes through (mid_x, k)
    # Total cost = score_left[k] + score_right[n - k]
    # Note: score_right is based on reversed strings, so index k corresponds to index n-k in forward string
    min_cost = float('inf')
    split_y = -1

    for k in range(n + 1):
        current_sum = score_left[k] + score_right[n - k]
        if current_sum < min_cost:
            min_cost = current_sum
            split_y = k

    # Recursively solve the two subproblems
    # Top-Left: X[0...mid_x] and Y[0...split_y]
    cost_tl, aligned_x_tl, aligned_y_tl = divide_and_conquer_alignment(X[:mid_x], Y[:split_y])
    
    # Bottom-Right: X[mid_x...end] and Y[split_y...end]
    cost_br, aligned_x_br, aligned_y_br = divide_and_conquer_alignment(X[mid_x:], Y[split_y:])

    # Combine results
    total_cost = cost_tl + cost_br
    total_aligned_x = aligned_x_tl + aligned_x_br
    total_aligned_y = aligned_y_tl + aligned_y_br

    return total_cost, total_aligned_x, total_aligned_y

# ===================== Main =====================

def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 1. Generate Strings
    X, Y = generate_strings_from_file(input_path)

    # 2. Run Efficient Alignment
    cost, aligned_X, aligned_Y, time_ms, mem_kb = run_with_measurement(
        divide_and_conquer_alignment, X, Y
    )

    # 3. Write Output
    with open(output_path, "w") as f:
        f.write(str(int(cost)) + "\n")
        f.write(aligned_X + "\n")
        f.write(aligned_Y + "\n")
        f.write(str(float(time_ms)) + "\n")
        f.write(str(float(mem_kb)) + "\n")

if __name__ == "__main__":
    main()