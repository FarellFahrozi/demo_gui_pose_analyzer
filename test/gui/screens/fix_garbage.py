
path = r"c:\Users\farellfahrozi\Documents\Projects\my_github\demo_gui_pose_analyzer\test\gui\screens\results.py"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Indicies are 0-based.
# We want to remove lines 1035 to 1238 (inclusive).
# Line 1035 is index 1034.
# Line 1238 is index 1237.
# We keep lines[:1034] and lines[1238:] (which is line 1239 onwards)

start_idx = 1034
end_idx = 1238

print(f"Removing lines {start_idx+1} to {end_idx} (inclusive)")
# Verify content
print(f"Start content: {lines[start_idx]}") # Should be "# Helper for dynamic xlim"
print(f"Content before retention: {lines[end_idx-1]}") # Should be "self.graph_figures.append(fig4)" or similar?
print(f"Content to retain (start of next): {lines[end_idx]}") # Should be "    def _draw_plumb_line..." or empty line?

if "def set_dynamic_xlim" not in lines[start_idx+1] and "# Helper" not in lines[start_idx]:
    print("WARNING: Start range content mismatch. Check indices.")
    print(f"Actual: {lines[start_idx]}")

if "def _draw_plumb_line" not in lines[end_idx+1] and "def _draw_plumb_line" not in lines[end_idx]:
     print("WARNING: End range content mismatch.")
     print(f"Actual at {end_idx}: {lines[end_idx]}")
     print(f"Actual at {end_idx+1}: {lines[end_idx+1]}")
     
new_lines = lines[:start_idx] + lines[end_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
    
print("Successfully removed garbage code.")
