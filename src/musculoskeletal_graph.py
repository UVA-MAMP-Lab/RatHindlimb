import opensim as osim
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from itertools import product
import logging
import math
import pandas as pd

class MusculoskeletalGraph: #TODO: Should probably just be a wrapper for the Model class
    def __init__(self, model_path: str | osim.Model, debug: bool = False, visualize: bool = False):

        self.model = osim.Model(model_path)
        self.model.initSystem()
        self.visualize = visualize
        self.model.setUseVisualizer(visualize)
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.console = logging.StreamHandler()
        self.log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Set logging level based on debug flag
        self.log_level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(self.log_level)
        self.console.setLevel(self.log_level)
        self.console.setFormatter(self.log_formatter)
        self.logger.addHandler(self.console)
        self.logger.debug(f"Model loaded from {model_path}")

        # Store relationships based on string names instead of object references
        self.muscle_names = []  # Muscle names
        self.joint_bodies: Dict[str, Tuple[str, str]] = {}  # Joint name -> (parent name, child name)
        self.bodies_joint: Dict[frozenset[str], str] = {}  # (parent name, child name) -> Joint name
        self.body_graph: Dict[str, Set[str]] = defaultdict(set)  # Adjacent body names
        self.muscle_attachments: Dict[str, Set[str]] = defaultdict(set)  # Muscle name -> body names
        self.muscle_wraps: Dict[str, Set[str]] = defaultdict(set)  # Muscle name -> wrap object names
        self.wraps_muscles: Dict[str, Set[str]] = defaultdict(set)  # Wrap object name -> muscle names
        self.body_wraps: Dict[str, Set[str]] = defaultdict(set)  # Body name -> wrap object names
        self.wrap_body: Dict[str, str] = {}  # Wrap object name -> body name
        self.path_cache: Dict[frozenset[str], List[str]] = {}  # Cached paths between body names
        self.muscle_crossings: Dict[str, Set[str]] = defaultdict(set)  # Muscle name -> crossed joint names
        self.crossings_muscle: Dict[frozenset[str], Set[str]] = defaultdict(set)  # Joint names -> muscle names
        self.muscle_coords: Dict[str, Set[str]] = defaultdict(set)  # Muscle name -> coordinate names
        self.coords_muscles: Dict[frozenset[str], Set[str]] = defaultdict(set)  # Coordinate names -> muscle names
        
        # Coordinate tracking
        self.joint_to_coords: Dict[str, Set[str]] = defaultdict(set)  # Joint name -> coordinate names
        self.coord_ranges: Dict[str, Tuple[float, float]] = {}  # Coordinate name -> (min, max)
        
        self._build_skeletal_graph()
        self._cache_muscle_attachments()
        self._cache_coordinate_ranges()
        self._cache_muscle_crossings()

    def get_muscle(self, muscle_name: str) -> Optional[osim.Muscle]:
        """Get a muscle by name."""
        muscles : osim.SetMuscles = self.model.getMuscles()
        try:
            return muscles.get(muscle_name)
        except:
            self.logger.error(f"Muscle {muscle_name} not found")
            return None
        
    def get_body(self, body_name: str) -> Optional[osim.Body]:
        """Get a body by name."""
        bodies : osim.BodySet = self.model.getBodySet()
        try:
            return bodies.get(body_name)
        except:
            self.logger.error(f"Body {body_name} not found")
            return None
    
    def get_joint(self, joint_name: str) -> Optional[osim.Joint]:
        """Get a joint by name."""
        joints : osim.JointSet = self.model.getJointSet()
        try:
            return joints.get(joint_name)
        except:
            return None
    
    def get_coordinate(self, coord_name: str) -> Optional[osim.Coordinate]:
        """Get a coordinate by name."""
        coords : osim.SetCoordinates = self.model.getCoordinateSet()
        try:
            return coords.get(coord_name)
        except:
            return None
    
    def get_wrap(self, wrap_name: str) -> Optional[osim.PathWrap]:
        """Get a wrap object by name."""
        body = self.wrap_body.get(wrap_name)
        if body:
            return self.get_body(body).getWrapObject(wrap_name) # getWrapObject says it's deprecated but it works
        return None

    def get_wrap_body(self, wrap_name: str) -> Optional[osim.Body]:
        """Get the body a wrap object is attached to."""
        return self.get_body(self.wrap_body.get(wrap_name))

    def get_muscles_wrapping(self, wrap_name: str) -> Set[osim.Muscle]:
        """Get all muscles that wrap a specific object."""
        return {self.get_muscle(muscle) for muscle in self.wraps_muscles.get(wrap_name, set())}

    def _build_skeletal_graph(self):
        """Builds the graph structure using joint relationships."""
        joints = self.model.getJointSet()
        
        for i in range(joints.getSize()):
            joint : osim.Joint = joints.get(i)
            # Get the actual bodies that the joint connects
            parent : osim.Frame = joint.getParentFrame().findBaseFrame()
            child : osim.Frame = joint.getChildFrame().findBaseFrame()
            
            parent_name = parent.getName()
            child_name = child.getName()
            joint_name = joint.getName()

            # Store the joint and its body connections
            self.joint_bodies[joint_name] = (parent_name, child_name)
            self.bodies_joint[frozenset([parent_name, child_name])] = joint_name
            
            # Add coordinates to the joint's set
            for j in range(joint.numCoordinates()):
                coord = joint.get_coordinates(j)
                self.joint_to_coords[joint_name].add(coord.getName())
            
            # Build undirected graph of body connections
            self.body_graph[parent_name].add(child_name)
            self.body_graph[child_name].add(parent_name)

            # Store body wrap objects
            try:
                parent_wrap_objects : osim.PathWrapSet = osim.Body.safeDownCast(parent).getWrapObjectSet()
                parent_wrap_names = set([wrap.getName() for wrap in parent_wrap_objects])
                self.body_wraps[parent_name] = parent_wrap_names
                for wrap_name in parent_wrap_names:
                    self.wrap_body[wrap_name] = parent_name
            except Exception as e:
                self.logger.error(f"Error processing parent wrap objects for {parent_name}: {e}")

            try:
                child_wrap_objects : osim.PathWrapSet = osim.Body.safeDownCast(child).getWrapObjectSet()
                child_wrap_names = set([wrap.getName() for wrap in child_wrap_objects])
                self.body_wraps[child_name] = child_wrap_names
                for wrap_name in child_wrap_names:
                    self.wrap_body[wrap_name] = child_name
            except Exception as e:
                self.logger.error(f"Error processing child wrap objects for {child_name}: {e}")

            self.logger.debug(f"Joint {joint_name} connects {parent_name} and {child_name}")

    def _cache_muscle_attachments(self):
        """Caches all muscle attachments using object references."""
        muscles : osim.SetMuscles = self.model.getMuscles()
        
        for i in range(muscles.getSize()):
            muscle : osim.Muscle = muscles.get(i)
            muscle_name = muscle.getName()
            path : osim.GeometryPath = muscle.getGeometryPath()
            
            # Store bodies this muscle attaches to
            path_points : osim.PathPointSet = path.getPathPointSet()
            attached_bodies = set()
            for j in range(path_points.getSize()):
                path_point : osim.PathPoint = path_points.get(j)
                frame : osim.PhysicalFrame = path_point.getParentFrame()
                body = frame.findBaseFrame()
                body_name = body.getName()
                attached_bodies.add(body_name)
            self.muscle_attachments[muscle_name] = attached_bodies

            # Store wrap objects
            path_wraps : osim.PathWrapSet = path.getWrapSet()
            wrap_objects = set()
            for j in range(path_wraps.getSize()):
                path_wrap : osim.PathWrap = path_wraps.get(j)
                wrap_object_name = path_wrap.getWrapObjectName()
                wrap_objects.add(wrap_object_name)
            for wrap_object in wrap_objects:
                self.wraps_muscles[wrap_object].add(muscle_name)
            self.muscle_wraps[muscle_name] = wrap_objects
        
            self.logger.debug(f"Muscle {muscle_name} attaches to bodies {attached_bodies} and wraps {wrap_objects}")
    
    def _cache_coordinate_ranges(self):
        """Cache coordinate information using object references."""
        coords  : osim.SetCoordinates = self.model.getCoordinateSet()
        
        for i in range(coords.getSize()):
            coord : osim.Coordinate = coords.get(i)
            coord_name = coord.getName()
            # Store coordinate ranges
            self.coord_ranges[coord_name] = (coord.getRangeMin(), coord.getRangeMax())

            self.logger.debug(f"Coordinate {coord_name} range: {self.coord_ranges[coord_name]}")

    def _find_path(self, bodies : List[osim.Frame]) -> List[osim.Frame]: # TODO: handle more than two bodies 
        """
        Finds the shortest path between two bodies using BFS.
        Uses a frozenset for order-independent caching of paths.

        Args:
            bodies: List of two or more bodies to find a path between.
        """
        if len(bodies) < 2:
            return []
        # First check the cache for a precomputed path
        start, end = bodies[0], bodies[-1]

        cache_key = frozenset([start, end])
        if cache_key in self.path_cache:
            path = self.path_cache[cache_key]
            # If the path starts with the end body, we need to reverse it
            return path if path[0] == start else path[::-1]

        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                self.path_cache[cache_key] = path
                self.logger.debug(f"Path found between {start} and {end}: {path}")
                return path
                
            for next_body in self.body_graph[current]:
                if next_body not in visited:
                    visited.add(next_body)
                    new_path = path + [next_body]
                    queue.append((next_body, new_path))
        
        self.logger.debug(f"No path found between {start.getName()} and {end.getName()}")
        return []
    
    def _cache_muscle_crossings(self): # TODO: handle case where muscle is attached to more than two bodies
        """Cache all joints crossed by each muscle."""
        for muscle_name, body_names in self.muscle_attachments.items():
            crossed_joints = set()
            
            path = self._find_path(list(body_names))
            for i in range(len(path) - 1):
                parent, child = path[i], path[i + 1]
                if joint := self.bodies_joint.get(frozenset([parent, child])): # Python 3.8+ because of the walrus operator
                    crossed_joints.add(joint)
            
            self.muscle_crossings[muscle_name] = crossed_joints
            self.crossings_muscle[frozenset(crossed_joints)].add(muscle_name)
            self.muscle_coords[muscle_name] = set().union(*(self.joint_to_coords[joint] for joint in crossed_joints))
            self.coords_muscles[frozenset(self.muscle_coords[muscle_name])].add(muscle_name)

            self.logger.debug(f"Muscle {muscle_name} crosses joints {crossed_joints}")

    def get_joints_for_muscle(self, muscle_name) -> List[osim.Joint]:
        """Get all joints crossed by a muscle."""
        return self.muscle_crossings.get(muscle_name, [])
        
    def get_muscle_length(self, muscle_name: str, state: osim.State) -> float:
        """Get the length of a muscle. Must realize position first."""
        return self.get_muscle(muscle_name).getLength(state)
        
    def get_muscle_lengths(self, muscle_names: List[str], state: osim.State) -> np.ndarray:
        """Get the lengths of multiple muscles. Must realize position first."""
        return np.array([self.get_muscle_length(name, state) for name in muscle_names])
        
    def get_coordinate_range(self, coord_name: str, res: int = 2) -> np.ndarray:
        """Get the values of a coordinate."""
        return np.linspace(*self.coord_ranges[coord_name], res)
    
    def get_coordinate_combinations(self, coordinates: List[str], res: int = 2) -> np.ndarray: # TODO: range for each coordinate
        """Get all possible combinations of coordinate values."""
        coordinate_values = [self.get_coordinate_range(coord, res) for coord in coordinates]
        return np.array(list(product(*coordinate_values)))
    
    def get_muscle_lengths_coordinates( 
        self,
        muscle_names: List[str],
        coordinates: List[str],
        min_points: int = 10,
    ) -> np.ndarray:
        """Analyze muscle length through the range of motion of multiple coordinates."""
        # Total points >= min_points = points per coordinate ^ number of coordinates
        points_per_coordinate = math.ceil(min_points ** (1 / len(coordinates)))
        coordinate_values = self.get_coordinate_combinations(coordinates, points_per_coordinate)
        state = self.model.initSystem()
        data = np.zeros((coordinate_values.shape[0], len(coordinates) + len(muscle_names)))
        data[:, :len(coordinates)] = coordinate_values
        for i, values in enumerate(coordinate_values):
            for coord, value in zip(coordinates, values):
                self.get_coordinate(coord).setValue(state, value)
            self.model.realizePosition(state)
            muscle_lengths = self.get_muscle_lengths(muscle_names, state)
            data[i, len(coordinates):] = muscle_lengths
        return data
    
    def muscle_rom_analysis(self, min_points: int = 10) -> dict[str, pd.DataFrame]:
        """
        Perform range of motion analyses based on the specified options.
        
        Returns:
            dict[str, pd.DataFrame]: A dictionary where keys are muscle names and values are 
            DataFrames containing analysis results, namely coordinate values and the optional 
            analyses.
        """
        
        results = {}
        for coord_set, muscles in self.coords_muscles.items():    
            state = self.model.initSystem()
            
            # check for locked coordinates
            unlocked_coords = set([coord for coord in coord_set if not self.get_coordinate(coord).getDefaultLocked()])            
            if not unlocked_coords:
                self.logger.warning(f"All coordinates in {coord_set} are locked for muscles {muscles}")
                continue
            
            # Iterate through coordinate combos
            points_per_coordinate = math.ceil(min_points ** (1 / len(unlocked_coords)))
            combos = self.get_coordinate_combinations(unlocked_coords, points_per_coordinate)
            
            # Prepare column names for all dataframes
            column_names = list(unlocked_coords) + ['length'] + ['moment_arm_'+coord for coord in unlocked_coords]
            
            # Loop through each muscle, creating a separate data array for each one
            for muscle_name in muscles:
               
                # Create a fresh DataFrame for this muscle
                results[muscle_name] = pd.DataFrame(np.zeros((combos.shape[0], len(column_names))), columns=column_names)
                results[muscle_name].loc[:, list(unlocked_coords)] = combos
                
                # Get this muscle
                muscle = self.get_muscle(muscle_name)
                
                # Compute lengths and moment arms for each coordinate combo
                for i, values in enumerate(combos):
                    for coord, value in zip(unlocked_coords, values):
                        self.get_coordinate(coord).setValue(state, value)
                
                    self.model.realizePosition(state)
                    
                    # Get length
                    length = muscle.getLength(state)
                    results[muscle_name].loc[i, 'length'] = length
                    
                    # Get moment arms
                    for coord_name in unlocked_coords:
                        coord = self.get_coordinate(coord_name)
                        moment_arm = muscle.computeMomentArm(state, coord)
                        results[muscle_name].loc[i, f'moment_arm_{coord_name}'] = moment_arm
                        
        return results

    def get_muscle_lengths_rom(
        self, 
        muscle_names: List[str], 
        min_points: int = 10, 
        max_length: Optional[float] = None, # TODO
        muscle_coords: Optional[List[str]] = None,
    ) -> pd.DataFrame: 
        """Analyze muscle length through the range of motion of all coordinates."""
        # TODO: Check for muscle crossings
        if muscle_coords is None:
            muscle_coords = list(set().union(*(self.muscle_coords[muscle] for muscle in muscle_names)))
        data = self.get_muscle_lengths_coordinates(muscle_names, muscle_coords, min_points)
        return pd.DataFrame(data, columns=muscle_coords + muscle_names)

    def get_all_muscle_lengths_rom(self, min_points: int = 10) -> dict[str, pd.DataFrame]:
        """
        Analyze muscle length through the range of motion of all coordinates.
        
        Returns:
            dict[str, pd.DataFrame]: A dictionary where keys are muscle names and 
            values are DataFrames containing coordinate values and muscle lengths.
        """
        # TODO: Subsets and/or parallelization to speed up computation
        results = {}
        for coord_set, muscles in self.coords_muscles.items():
            # check for locked coordinates
            unlocked_coords = set([coord for coord in coord_set if not self.get_coordinate(coord).getDefaultLocked()])
            if not unlocked_coords:
                self.logger.warning(f"All coordinates in {coord_set} are locked for muscles {muscles}")
                continue
            diff = coord_set.difference(unlocked_coords)
            if diff:
                self.logger.warning(f"Locked coordinates {diff} for muscles {muscles}")
            df = self.get_muscle_lengths_rom(list(muscles), muscle_coords=list(unlocked_coords), min_points=min_points)
            # Add the coordinate values and muscle lengths to the results dictionary for each muscle
            results.update({muscle: df[list(unlocked_coords) + [muscle]] for muscle in muscles})
        
        self.logger.debug(f"Calculated muscle lengths results: {results}")
        
        return results