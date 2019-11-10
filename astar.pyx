cimport numpy as np
import heapq
import numpy as np
cimport cython

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def find_path(grid, start, goal):
    '''
    Function recieves a numpy 2d array, 
            index of start point and index of end point
    Function returnes path - a list of indexes, 
            representing the shortest path from start cell to end cell
    '''
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    frontier_ = []
    cdef long frontier_len = 0

    heapq.heappush(frontier_, (0, start))
    frontier_len += 1

    cdef long neighbors_i[4]
    cdef long neighbors_j[4]
    cdef int i

    cdef long size_i, size_j
    cdef long current_i, current_j
    cdef long next_i, next_j
    cdef long goal_i, goal_j
    cdef long next_cost
    cdef long new_cost
    cdef long hc, priority
    cdef long [:,::1] grid_ = grid

    goal_i, goal_j = goal
    size_i, size_j = grid.shape

    cost_sf_ = np.zeros((size_i,size_j), dtype=np.dtype("l"))
    cost_sf_ -= 1
    cdef long [:,::1] cost_so_far_ = cost_sf_
    cost_so_far_[start[0],start[1]]=0


    while frontier_len > 0:
        current = heapq.heappop(frontier_)[1]
        frontier_len -=1
        
        current_i,current_j = current

        if current_i == goal_i and current_j == goal_j:
            break

        get_neighbors2(size_i, size_j,
                       current_i,current_j,
                       neighbors_i, neighbors_j)

        # walk over 4 neighbors
        for i in range(4):
            next_i = neighbors_i[i]
            next_j = neighbors_j[i]

            new_cost = cost_so_far_[current_i, current_j] + \
		                         get_cost(grid_, 
                                         current_i, current_j, 
                                         next_i, next_j)

            next_cost = cost_so_far_[next_i, next_j]
            if next_cost == -1 or new_cost < next_cost:
                cost_so_far_[next_i,next_j] = new_cost
                hc = heuristic(goal_i, goal_j, next_i, next_j)
                priority = new_cost + hc
                next = next_i, next_j
                #frontier.put(next, priority)
                heapq.heappush(frontier_, (priority, next))
                frontier_len += 1
                came_from[next] = current
    
    path = reconstruct_path(came_from, start, goal)
    return path

cdef long get_cost(long[:, ::1] grid, 
                     long current_i, long current_j, 
                     long next_i, long next_j):
    return grid[next_i,next_j]

cdef int get_neighbors2(long size_i, long size_j, 
                      long current_i, long current_j,
                   long neighbors_i[], long neighbors_j[]):

    if current_i > size_i - 1: return 0
    if current_j > size_j - 1: return 0
    if current_i < 0: return 0
    if current_j < 0: return 0

    cdef int i
    for i in range(4):
        neighbors_i[i] = -1
        neighbors_j[i] = -1

    if current_i > 0: 
        neighbors_i[0] = current_i-1
        neighbors_j[0] = current_j
    if current_j > 0: 
        neighbors_i[1] = current_i
        neighbors_j[1] = current_j-1
    if current_i < size_i - 1: 
        neighbors_i[2] = current_i+1
        neighbors_j[2] = current_j
    if current_j < size_j - 1: 
        neighbors_i[3] = current_i
        neighbors_j[3] = current_j+1
    return 1


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
    cdef long current_i, current_j
    cdef long start_i, start_j

    current = goal
    start_i, start_j = start
    current_i, current_j = goal

    path = []
    while True:
        if current_i == start_i and current_j == start_j:
            break
        path.append((current_i, current_j))
        current_i, current_j = came_from[(current_i, current_j)]
    path.append((start_i, start_j)) # optional
    path.reverse() # optional
    return path
