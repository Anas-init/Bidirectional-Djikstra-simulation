import tkinter as tk
from tkinter import simpledialog, messagebox, colorchooser, ttk
import heapq
import math
import networkx as nx
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AdvancedGraphVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Bidirectional Dijkstra Visualizer")
        self.root.geometry("1200x800")

        self.graph = nx.Graph()
        self.pos = {}
        self.source = None
        self.target = None
        
        self.setup_ui()
        self.setup_canvas_interaction()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, relief=tk.RAISED, borderwidth=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        buttons = [
            ("Add Node", self.prepare_add_node),
            ("Add Edge", self.prepare_add_edge),
            ("Set Source", self.set_source),
            ("Set Target", self.set_target),
            ("Find Shortest Path", self.run_bidirectional_dijkstra),
            ("Clear Graph", self.clear_graph),
            ("Save Graph", self.save_graph),
            ("Load Graph", self.load_graph)
        ]

        for text, command in buttons:
            btn = tk.Button(control_frame, text=text, command=command, width=20)
            btn.pack(pady=5)
        self.status_var = tk.StringVar()
        status_label = tk.Label(control_frame, textvariable=self.status_var, wraplength=200)
        status_label.pack(pady=10)

    def setup_canvas_interaction(self):
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.add_node_mode = False
        self.add_edge_mode = False
        self.edge_start = None

    def on_canvas_click(self, event):
        if not event.inaxes:
            return

        if self.add_node_mode:
            name = f"N{len(self.graph.nodes) + 1}"
            color = colorchooser.askcolor(title="Choose node color")[1] or "skyblue"
            self.graph.add_node(name, color=color)
            self.pos[name] = (event.xdata, event.ydata)
            self.add_node_mode = False
            self.draw_graph()
            self.status_var.set(f"Node {name} added")

        elif self.add_edge_mode:
            if not self.pos:
                return
            
            nearest_node = min(self.pos.keys(), 
                                key=lambda n: ((self.pos[n][0]-event.xdata)**2 + 
                                               (self.pos[n][1]-event.ydata)**2)**0.5)
            
            if self.edge_start is None:
                self.edge_start = nearest_node
                self.status_var.set(f"Selected start node: {nearest_node}")
            else:
                end_node = nearest_node
                if self.edge_start != end_node:
                    weight = simpledialog.askfloat("Add Edge", f"Weight of edge {self.edge_start} -> {end_node}")
                    if weight is not None:
                        self.graph.add_edge(self.edge_start, end_node, weight=weight)
                        self.draw_graph()
                        self.status_var.set(f"Edge added between {self.edge_start} and {end_node}")
                
                self.edge_start = None
                self.add_edge_mode = False

    def prepare_add_node(self):
        self.add_node_mode = True
        self.add_edge_mode = False
        self.status_var.set("Click on canvas to add node")

    def prepare_add_edge(self):
        if len(self.graph.nodes) < 2:
            messagebox.showerror("Error", "Add more nodes first!")
            return
        
        self.add_edge_mode = True
        self.add_node_mode = False
        self.edge_start = None
        self.status_var.set("Click to select start node for edge")

    def set_source(self):
        nodes = list(self.graph.nodes)
        if nodes:
            self.source = simpledialog.askstring("Set Source", "Choose source node:", initialvalue=nodes[0])
            self.status_var.set(f"Source set: {self.source}")
            self.draw_graph()

    def set_target(self):
        nodes = list(self.graph.nodes)
        if nodes:
            self.target = simpledialog.askstring("Set Target", "Choose target node:", initialvalue=nodes[0])
            self.status_var.set(f"Target set: {self.target}")
            self.draw_graph()




    def run_bidirectional_dijkstra(self):
        if not self.source or not self.target:
            messagebox.showerror("Error", "Set source and target nodes!")
            return

        try:
            def bidirectional_dijkstra(graph, source, target):
                """Perform Bidirectional Dijkstra Algorithm."""
                start_total_time = time.perf_counter()  # Total algorithm start time
                
                # Measure initialization time
                start_init_time = time.perf_counter()
                if source == target:
                    return [source], 0, [], [], {
                        'total_time': 0,
                        'init_time': 0,
                        'forward_search_time': 0,
                        'backward_search_time': 0,
                        'path_reconstruction_time': 0
                    }
                
                forward_queue = [(0, source)]
                backward_queue = [(0, target)]

                forward_dist = {source: 0}
                backward_dist = {target: 0}

                forward_visited = {}
                backward_visited = {}

                meeting_node = None
                best_path_length = float("inf")

                forward_edges = []
                backward_edges = []
                
                init_time = time.perf_counter() - start_init_time

                # Measure search time
                start_search_time = time.perf_counter()
                forward_time = 0
                backward_time = 0

                forward_iterations = 0
                backward_iterations = 0

                while forward_queue and backward_queue:
                    # Forward search
                    start_forward_time = time.perf_counter()
                    f_dist, f_node = heapq.heappop(forward_queue)
                    if f_node in forward_visited:
                        forward_time += time.perf_counter() - start_forward_time
                        continue
                    forward_visited[f_node] = f_dist

                    for neighbor, attrs in graph[f_node].items():
                        weight = attrs.get("weight", 1)
                        new_dist = f_dist + weight
                        if neighbor not in forward_dist or new_dist < forward_dist[neighbor]:
                            forward_dist[neighbor] = new_dist
                            forward_edges.append((f_node, neighbor))
                            heapq.heappush(forward_queue, (new_dist, neighbor))

                    if f_node in backward_visited:
                        total_distance = f_dist + backward_visited[f_node]
                        if total_distance < best_path_length:
                            best_path_length = total_distance
                            meeting_node = f_node
                    
                    forward_iterations += 1
                    forward_time += time.perf_counter() - start_forward_time

                    # Backward search
                    start_backward_time = time.perf_counter()
                    b_dist, b_node = heapq.heappop(backward_queue)
                    if b_node in backward_visited:
                        backward_time += time.perf_counter() - start_backward_time
                        continue
                    backward_visited[b_node] = b_dist
                    
                    for neighbor, attrs in graph[b_node].items():
                        weight = attrs.get("weight", 1)
                        new_dist = b_dist + weight
                        if neighbor not in backward_dist or new_dist < backward_dist[neighbor]:
                            backward_dist[neighbor] = new_dist
                            backward_edges.append((b_node, neighbor))
                            heapq.heappush(backward_queue, (new_dist, neighbor))
                    
                    if b_node in forward_visited:
                        total_distance = b_dist + forward_visited[b_node]
                        if total_distance < best_path_length:
                            best_path_length = total_distance
                            meeting_node = b_node
                    
                    backward_iterations += 1
                    backward_time += time.perf_counter() - start_backward_time

                if meeting_node is None:
                    raise nx.NetworkXNoPath
                
                # Reconstruct path
                start_path_time = time.perf_counter()
                path = []
                current = meeting_node
                while current in forward_visited:
                    path.append(current)
                    current = next((n for n in graph[current] if n in forward_visited and
                                    forward_visited[n] + graph[current][n]["weight"] == forward_visited[current]), None)
                path.reverse()

                current = meeting_node
                while current in backward_visited:
                    current = next((n for n in graph[current] if n in backward_visited and
                                    backward_visited[n] + graph[current][n]["weight"] == backward_visited[current]), None)
                    if current:
                        path.append(current)
                path_time = time.perf_counter() - start_path_time

                # Calculate total time
                total_time = time.perf_counter() - start_total_time

                return path, best_path_length, forward_edges, backward_edges, {
                    'total_time': total_time,
                    'init_time': init_time,
                    'forward_search_time': forward_time,
                    'backward_search_time': backward_time,
                    'path_reconstruction_time': path_time,
                    'forward_iterations': forward_iterations,
                    'backward_iterations': backward_iterations
                }

            def animate_path_discovery():
                # Capture detailed timing information
                path, path_length, forward_edges, backward_edges, timing_info = bidirectional_dijkstra(self.graph, self.source, self.target)

                # Construct detailed timing status message
                timing_status = (
                    f"Total Algorithm Time: {timing_info['total_time']:.6f} seconds\n"
                    f"Initialization Time: {timing_info['init_time']:.6f} seconds\n"
                    f"Forward Search Time: {timing_info['forward_search_time']:.6f} seconds\n"
                    f"Backward Search Time: {timing_info['backward_search_time']:.6f} seconds\n"
                    f"Path Reconstruction Time: {timing_info['path_reconstruction_time']:.6f} seconds\n"
                    f"Forward Search Iterations: {timing_info['forward_iterations']}\n"
                    f"Backward Search Iterations: {timing_info['backward_iterations']}\n"
                    f"Path: {' -> '.join(path)}\n"
                    f"Path Length: {path_length}"
                )
                
                # Update status variable with detailed timing information
                self.status_var.set(timing_status)

                # Rest of the animation code remains the same
                steps = max(len(forward_edges), len(backward_edges))
                for step in range(steps):
                    self.ax.clear()
                    node_colors = [self.graph.nodes[n].get('color', 'skyblue') for n in self.graph.nodes()]
                    nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, ax=self.ax)
                    nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax)
                    nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax)
                    if step < len(forward_edges):
                        nx.draw_networkx_edges(self.graph, self.pos, edgelist=forward_edges[:step + 1],
                                            edge_color='Red', width=2, ax=self.ax, label="Forward Search")
                    if step < len(backward_edges):
                        nx.draw_networkx_edges(self.graph, self.pos, edgelist=backward_edges[:step + 1],
                                            edge_color='Orange', width=2, ax=self.ax, label="Backward Search")

                    nx.draw_networkx_edges(self.graph, self.pos, edgelist=list(zip(path, path[1:])),
                                        edge_color='red', width=3, ax=self.ax, label="Final Path")
                    edge_labels = nx.get_edge_attributes(self.graph, 'weight')
                    nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, ax=self.ax)

                    self.ax.set_title(f"Step {step + 1}: Forward (Red), Backward (orange)")
                    self.canvas.draw()
                    self.root.update()
                    self.root.after(500)

            animate_path_discovery()

        except nx.NetworkXNoPath:
            messagebox.showinfo("Result", "No path exists between nodes!")
    def draw_graph(self, path=None):
        self.ax.clear()
        node_colors = [self.graph.nodes[n].get('color', 'skyblue') for n in self.graph.nodes()]
        
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, ax=self.ax)
            nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax)
            nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax)
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=path_edges, edge_color='r', width=2, ax=self.ax)
        else:
            nx.draw(self.graph, self.pos, with_labels=True, node_color=node_colors, ax=self.ax)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, ax=self.ax)

        self.ax.set_title("Graph Visualization")
        self.canvas.draw()

    def clear_graph(self):
        self.graph.clear()
        self.pos.clear()
        self.source = None
        self.target = None
        self.status_var.set("Graph cleared")
        self.draw_graph()

    def save_graph(self):
        filename = simpledialog.askstring("Save Graph", "Enter filename:")
        if filename:
            nx.write_graphml(self.graph, f"{filename}.graphml")
            self.status_var.set(f"Graph saved to {filename}.graphml")

    def load_graph(self):
        filename = simpledialog.askstring("Load Graph", "Enter filename:")
        if filename:
            try:
                self.graph = nx.read_graphml(f"{filename}.graphml")
                self.pos = nx.spring_layout(self.graph)
                self.draw_graph()
                self.status_var.set(f"Graph loaded from {filename}.graphml")
            except FileNotFoundError:
                messagebox.showerror("Error", "File not found!")

def main():
    root = tk.Tk()
    app = AdvancedGraphVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()