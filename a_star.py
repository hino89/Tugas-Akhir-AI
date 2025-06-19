import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import random

# === Generate Random Maze 20x20 (0 = jalan, 1 = tembok) ===
rows, cols = 20, 20
# random.seed(42)
maze = [[0 if random.random() > 0.25 else 1 for _ in range(cols)] for _ in range(rows)]

start = (0, 0)
goal = (rows - 1, cols - 1)
maze[start[0]][start[1]] = 0
maze[goal[0]][goal[1]] = 0

# === A* Algorithm ===
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()
    visited_order = []

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        visited_order.append(current)

        if current == goal:
            return path, visited_order

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    heapq.heappush(open_set, (g + 1 + heuristic(neighbor, goal), g + 1, neighbor, path + [neighbor]))

    return None, visited_order

# === Jalankan A* ===
path, visited_order = astar(maze, start, goal)

# === Setup Visualisasi ===
grid = [[1 if cell == 1 else 0 for cell in row] for row in maze]  # 1 = tembok, 0 = jalan
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap='viridis')

def update(i):
    if i < len(visited_order):
        x, y = visited_order[i]
        if (x, y) != start and (x, y) != goal:
            grid[x][y] = 0.5  # dikunjungi = hijau muda
    elif path and i < len(visited_order) + len(path):
        x, y = path[i - len(visited_order)]
        if (x, y) != start and (x, y) != goal:
            grid[x][y] = 0.2  # jalur = ungu tua

    # Tandai start dan goal dengan nilai khusus
    sx, sy = start
    gx, gy = goal
    grid[sx][sy] = 0.8  # start = kuning
    grid[gx][gy] = 0.1  # goal = biru tua

    im.set_data(grid)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(visited_order)+(len(path) if path else 0), interval=50, repeat=False)
plt.title("A* Maze Solver (20x20)")
plt.show()
