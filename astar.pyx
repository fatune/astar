cimport numpy as np
import heapq

def find_path(grid, start, goal):
    '''
    Function recieves a numpy 2d array, 
            index of start point and index of end point
    Function returnes path - a list of indexes, 
            representing the shortest path from start cell to end cell
    '''
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        neighbors = get_neighbors(grid, current)

        for next in neighbors:
            new_cost = cost_so_far[current] + get_cost(grid, 
                                         current[0], current[1], 
                                         next[0], next[1])
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                hc = heuristic(goal[0], goal[1], next[0], next[1])
                priority = new_cost + hc
                frontier.put(next, priority)
                came_from[next] = current
    
    path = reconstruct_path(came_from, start, goal)
    return path

cdef double get_cost(double[:, :] grid, 
                     long current_i, long current_j, 
                     long next_i, long next_j):
    return grid[next_i,next_j]


def get_neighbors(grid, current):
    """
    returns possible neighbors of the current cell
    maximum possible neighbors is 4
    """
    neighbors = []

    i, j = current
    size_i, size_j = grid.shape

    if i > size_i - 1: return []
    if j > size_j - 1: return []
    if i < 0: return []
    if j < 0: return []

    if i > 0: neighbors.append((i-1,j))
    if j > 0: neighbors.append((i,j-1))
    if i < size_i - 1: neighbors.append((i+1,j))
    if j < size_j - 1: neighbors.append((i,j+1))

    return neighbors

cdef long heuristic(long x1, long y1, long x2, long y2):
    return abs(x1 - x2) + abs(y1 - y2)


class PriorityQueue:
    def __init__(self):
        self.elements = []
    		    
    def empty(self):
        return len(self.elements) == 0
    				    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    	
    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path
