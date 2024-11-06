"""
Path class, for doing operations on paths

Clay Foye, sep 2024
"""


class Path:
    """
    Path Class

    This class handles different paths, including storing their properties, length, etc.

    Later on, we can extend this class to do operations on paths, or even generate new paths.
    For example, we can write algos to check:
        - whether or not this path has a cycle of length n
        - the difference from the optimal path between the origin and end
        - render the path


    Properties
    ----------
    path : list of str
        A list of article titles. Can eventually be article objects?
    origin : str
        The beginning of the path
    end : str
        The end of the path
    length : int
        the length of the path, in nodes
    
    

    """