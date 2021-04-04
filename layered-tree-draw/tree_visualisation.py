import copy
import dataclasses
import typing as tp

import networkx as nx

from matplotlib import pyplot as plt


def _find_roots(tree: nx.DiGraph) -> tp.List[tp.Hashable]:
    """
    Найти все корни (вершины нулевой входящей степени) 
    у ориентированного дерева.
    """
    roots = []
    for node in tree.nodes():
        if tree.in_degree(node) == 0:
            roots.append(node)
    return roots


def _get_basic_layered_layout(
    tree: nx.DiGraph, root: tp.Hashable
) -> tp.Dict[tp.Hashable, tp.Tuple[float, float]]:
    """
    Простейший алгоритм укладки дерева на основе in-order обхода.
    Работает для произвольных деревьев, не только бинарных.
    
    Вычисляет укладку вида (in_order, -depth) и возвращает её в виде
    словаря (ключ, координаты) -- стандартном для networkx формате.
    """
    
    layout: tp.Dict[tp.Hashable, tp.Tuple[float, float]] = {}
    
    def _recursive_subroutine(node: tp.Hashable, depth: int) -> int:
        """
        In-order обход дерева, который заполняет layout.
        """
        children = list(nx.neighbors(tree, node))
        inorder_id = 0
        for child in children:
            inorder_id = max(
                inorder_id, 
                _recursive_subroutine(child, depth + 1)
            )
        inorder_id += 1
        layout[node] = (inorder_id, -depth)
        return inorder_id
        
    _ = _recursive_subroutine(root, 0)
    return layout


def _center_layered_layout(
    tree: nx.DiGraph, 
    root: tp.Hashable, 
    layout: tp.Dict[tp.Hashable, tp.Tuple[float, float]]
) -> tp.Dict[tp.Hashable, tp.Tuple[float, float]]:
    """
    Центрирует укладку по X-координате и возвращает её в виде
    словаря (ключ, координаты) -- стандартном для networkx формате.
    Может работать с произвольными, не обязательно бинарными деревьями.
    """
    
    refined_layout: tp.Dict[tp.Hashable, tp.Tuple[float, float]] = {}
    
    
    def _recursive_subroutine(node: tp.Hashable):
        children = list(nx.neighbors(tree, node))
        for child in children: 
            _recursive_subroutine(child)
            
        children_x_coords = [
            refined_layout[child][0] for child in children
        ]
        refined_x_coord = (
            layout[node][0] 
            if not children 
            else (min(children_x_coords) + max(children_x_coords)) / 2
        )
        refined_y_coord = layout[node][1] 
        refined_layout[node] = (refined_x_coord, refined_y_coord)
                    
    _recursive_subroutine(root)
    return refined_layout


"""
Reingold-Tilford Layout
"""


@dataclasses.dataclass
class TreeContour:
    left_contour: tp.List[str]
    right_contour: tp.List[str]
        
        
def merge_contours(
    root_node: str,
    left_subtree_contour: TreeContour, 
    right_subtree_contour: TreeContour
) -> TreeContour:
    """
    Слить контуры поддеревьев (с учётом возможной разницы в высотах).
    """
    new_left_contour = (
        [root_node] 
        + left_subtree_contour.left_contour
        + right_subtree_contour.left_contour[
            len(left_subtree_contour.left_contour):
        ]
    )
    
    new_right_contour = (
        [root_node]
        + right_subtree_contour.right_contour
        + left_subtree_contour.right_contour[
            len(right_subtree_contour.right_contour):
        ]
    )
    
    return TreeContour(
        left_contour=new_left_contour,
        right_contour=new_right_contour
    )
    

def _ensure_margins_between_vertices_of_the_same_depth(
    tree: nx.DiGraph, 
    root: tp.Hashable, 
    layout: tp.Dict[tp.Hashable, tp.Tuple[float, float]],
    min_allowed_margin: float = 1
) -> tp.Dict[tp.Hashable, tp.Tuple[float, float]]:
    """
    Видоизменить in-order укладку таким образом, чтобы расстояние 
    между вершинами на одной глубине было не меньше min_allowed_margin.
    
    !!! Работает только для бинарных деревьев. !!!
    """
    
    x_offset: tp.Dict[tp.Hashable, float] = {root: 0}
    
    
    def _make_x_offsets_relative(node: str) -> None:
        """
        Релятивизация in-order укладки: вычисляем сдвиг каждого
        потомка относительно родителя по X-координате
        """
        for child in list(nx.neighbors(tree, node)):
            x_offset[child] = layout[child][0] - layout[node][0]
            _make_x_offsets_relative(child)
            
    _make_x_offsets_relative(root)
    
    
    def _recursive_subroutine(node: str) -> TreeContour:
        """
        Вычисляем контуры, а если те слишком близко -- увеличиваем зазор.
        """
        children = list(nx.neighbors(tree, node))
        if not children:
            return TreeContour(left_contour=[node], right_contour=[node])
        
        # Если будет время, обобщу алгоритм на случай произвольных деревьев
        
        leftmost_child = children[0]
        rightmost_child = children[-1]
        
        # Рекурсиво обрабатываем поддеревья
        
        left_subtree_contour = _recursive_subroutine(leftmost_child)
        
        # Если потомок всего один, то ничего не сдвигаем
        
        if leftmost_child == rightmost_child:
            return merge_contours(
                node, left_subtree_contour, left_subtree_contour
            )
        
        right_subtree_contour = _recursive_subroutine(rightmost_child)
        
        # Иначе смотрим на зазор на каждом уровне, находим минимум, правим сдвиги
        
        left_offset = 0
        right_offset = 0
        min_margin = float("inf")
        
        for left_node, right_node in zip(
            left_subtree_contour.right_contour, 
            right_subtree_contour.left_contour
        ):
            # Считаем честные координаты вершин
            left_offset += x_offset[left_node]
            right_offset += x_offset[right_node]
            
            # и честный зазор (с учётом уже имеющихся правок в поддеревьях)
            min_margin = min(min_margin, right_offset - left_offset)

        # Если зазор меньше минимально допустимого, то сдвигаем правое поддерево
        if min_margin < min_allowed_margin:
            additional_offset = min_allowed_margin - min_margin
            x_offset[rightmost_child] += additional_offset

        return merge_contours(
            node, left_subtree_contour, right_subtree_contour
        )
        
    _ = _recursive_subroutine(root)
    
    refined_layout: tp.Dict[tp.Hashable, tp.Tuple[float, float]] = {root: (0, 0)}
        
    
    def _compute_refined_layout(node: str) -> None:
        """
        Вычисляем координаты укладки по промежуточным сдвигам.
        """
       
        for child in list(nx.neighbors(tree, node)):
            refined_layout[child] = (
                x_offset[child] + refined_layout[node][0],
                layout[child][1]
            )
            _compute_refined_layout(child)
            
    _compute_refined_layout(root)
    return refined_layout


def get_layered_layout(
    binary_tree: nx.DiGraph, min_allowed_margin: float = 1
) -> tp.Dict[tp.Hashable, tp.Tuple[float, float]]:
    if not nx.is_tree(binary_tree):
        raise nx.NotATree()
        
    assert all(
        binary_tree.out_degree(node) <= 2 for node in binary_tree.nodes()
    ), "Binary tree expected!"
    
    roots = _find_roots(binary_tree)
    assert len(roots) == 1, "Rooted tree expected!"
    root = roots[0]
    
    layout = _get_basic_layered_layout(binary_tree, root)
    layout = _ensure_margins_between_vertices_of_the_same_depth(
        binary_tree, root, layout, min_allowed_margin
    )
    layout = _center_layered_layout(binary_tree, root, layout)
    return layout
