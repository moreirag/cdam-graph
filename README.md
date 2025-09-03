# cdam-graph
CDAM–Graph: A Unified Visualization Tool for High-Dimensional Data

Use CDAM-graph to visualize points in n-dimensional Euclidean space. The data should be in a .csv file, where each row represents a point and the columns represent coordinates. The file and its location should be specified in line 101 of the 'CDAM.py' script.

# How to read the graph

The graph is divided into circular sectors representing the Angular Mapping and a central sector representing the Chord Diagram.

## Angular Mapping

For points in an n-dimensional space, there are n sectors. The i-th sector maps the region of space closest to the i-th coordinate axis, considering the smallest angular distance between the point and the axes of the canonical basis of the space. For example, a point mapped in the first sector is located in the region of space closest to the first coordinate axis of the canonical basis. To better understand the representation, let's interpret each point as a vector.

In each sector, there are two axes. The horizontal axis represents the smallest angle between the vector and its corresponding axis. The vertical axis represents the vector's norm.

## Chord Diagram

The central region of the graph displays the Chord Diagram of the points. Similar to Angular Mapping, the Chord Diagram is divided into arcs. Each arc represents the interval where the coordinates of the points lie. For example, for the vector v=(2, 1, 3) in three-dimensional space, there will be three arcs, and the markings will be 2 on the first arc, 1 on the second arc, and 3 on the third arc. The markings are connected by Bézier curves within the chord diagram.
