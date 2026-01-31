from typing import List, Dict

class Node:
    def __init__(self, name: str, dims: list) -> None:
        self._name = name
        self._dims = dims
        self.edges: List['Edge'] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> list:
        return self._dims

    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)
            
    # [NEW] 엣지 제거 메서드
    def remove_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)

class Edge:
    def __init__(self, node0: Node, node1: Node) -> None:
        self._node0 = node0
        self._node1 = node1
        self._messages = {} 
        
        node0.add_edge(self)
        node1.add_edge(self)

    def get_other(self, node: Node) -> Node:
        return self._node1 if node is self._node0 else self._node0

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def connect(self, node0: Node, node1: Node):
        edge = Edge(node0, node1)
        self.edges.append(edge)
        if node0 not in self.nodes: self.nodes.append(node0)
        if node1 not in self.nodes: self.nodes.append(node1)
        return edge

    # [NEW] 엣지 삭제 (양쪽 노드에서 끊고, 그래프 목록에서 제거)
    def remove_edge(self, edge: Edge):
        if edge in self.edges:
            edge._node0.remove_edge(edge)
            edge._node1.remove_edge(edge)
            self.edges.remove(edge)

    # [NEW] 노드 삭제 (연결된 모든 엣지를 끊고 노드 제거)
    def remove_node(self, node: Node):
        if node in self.nodes:
            # 1. 연결된 모든 엣지 먼저 제거
            # (리스트를 순회하면서 삭제하므로 list()로 복사본을 만들어 순회)
            for edge in list(node.edges):
                self.remove_edge(edge)
            
            # 2. 노드 목록에서 제거
            self.nodes.remove(node)