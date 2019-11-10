import numpy as np

from astar import find_path
#from astar import get_neighbors

from astar_pure_python import find_path as find_path_pure_python

def test_compare_time():
    from time import time

    grid = np.zeros((1000,1000), dtype=np.dtype("l"))
    start = (1,1)
    goal = (903,901)
    
    time0 = time()
    result1 = find_path(grid,start,goal)
    time_cython = time()-time0
    time1 = time()
    result2 = find_path_pure_python(grid,start,goal)
    time_pure = time()-time1
    assert result1 == result2
    print ("cython: %s", time_cython)
    print ("pure py: %s", time_pure)


def test_function_returns_list():
    grid = np.zeros((5,10),dtype=np.dtype("l"))
    start = (1,1)
    goal = (3,1)
    result = find_path(grid,start,goal)
    assert result != 1
    assert type(result) is list
    assert result == [(1, 1), (2, 1), (3, 1)]

    goal = (3,3)
    result = find_path(grid,start,goal)
    assert result == [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]

    start = (1,1)
    goal = (3,1)
    grid[2,1] = 10
    result = find_path(grid,start,goal)
    print(result)

#def test_function_get_neighbors():
#    grid = np.zeros((5,10))
#
#    result = get_neighbors(grid, (0,0))
#    assert result == [(1,0),(0,1)]
#
#    result = get_neighbors(grid, (1,1))
#    assert result == [(0,1),(1,0),(2,1),(1,2)]
#
#    result = get_neighbors(grid, (0,2))
#    assert result == [(0,1), (1,2), (0,3)]
#
#    result = get_neighbors(grid, (100,100))
#    assert result == []
#
#    result = get_neighbors(grid, (-1,-1))
#    assert result == []
#
#    result = get_neighbors(grid, (5,10))
#    assert result == []
#
#    result = get_neighbors(grid, (4,9))
#    assert result == [(3,9),(4,8)]
#    
