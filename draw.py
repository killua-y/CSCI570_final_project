import matplotlib.pyplot as plt

# Test data
problem_size = [16, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 3968]
time_efficient = [0.66, 2.55, 6.35, 12.77, 29.51, 50.01, 112.27, 170.27, 277.95, 364.18, 630.27, 994.02, 1403.15, 1838.10, 2275.87]
time_basic = [0.19, 0.63, 1.62, 6.13, 14.29, 25.62, 59.53, 111.81, 176.05, 228.30, 409.43, 647.83, 927.92, 1327.98, 1675.70]
# mem_efficient = [0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 256, 256, 256, 384]
# mem_basic = [0, 0, 128, 640, 1408, 2560, 5888, 10368, 16128, 23296, 41216, 64384, 92800, 126208, 154752]


# Create the plot
plt.figure(figsize=(10,6))
plt.plot(problem_size, time_efficient, marker='o', color='blue', label='Efficient Algorithm')
plt.plot(problem_size, time_basic, marker='s', color='red', label='Basic Algorithm')
# plt.plot(problem_size, mem_efficient, marker='o', color='blue', label='Efficient Algorithm')
# plt.plot(problem_size, mem_basic, marker='s', color='red', label='Basic Algorithm')

# Labels and title
plt.xlabel('Problem Size')

plt.ylabel('Time(ms)')
plt.title('CPU Time vs Problem Size')
# plt.ylabel('Memory(KB)')
# plt.title('MEM Usage vs Problem Size')

plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('time_comparison.png')
# plt.savefig('memory_comparison.png')
# Show plot
plt.show()


