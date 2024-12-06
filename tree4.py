import math
from enum import Enum
import numpy as np
class NodeType(Enum):
    B_OP=1,
    U_OP=2,
    VAR=3,
    CONST=4
class Node:

    def __init__(self,data, node_type:NodeType):
        self.data = data
        self.type = node_type


class FastTBTree:
    def __init__(self, md:int,randomize=False):
        # self.tree:np.ndarray = [None] *2**md - 1
        # self.tree:np.ndarray = np.array([randomize] * (2**md - 1))
        if randomize:
            self.tree = np.random.randint(0, 100, size=(2**md - 1))
        else:
            self.tree = np.array([None] * (2**md - 1))
        self.md = md

    def set_params(operators,n_vars,max_const):
        FastTBTree.operators = operators
        FastTBTree.n_vars = n_vars
        FastTBTree.max_const = max_const
    def print_tree(self):
        # check if np array is empty
        if len(self.tree) == 0:
            print("<empty self.tree>")
            return
        
        height = math.ceil(math.log2(len(self.tree) + 1))  # Height of tree
        max_width = 2 ** (height - 1)  
        
        level = 0
        index = 0
        while index < len(self.tree):
            level_nodes = 2 ** level
            spacing = " " * ((max_width // (level_nodes + 1))+1)*5
            line = spacing.join(
                str(self.tree[index + i]) if index + i < len(self.tree) and self.tree[index + i] is not None else ""
                for i in range(level_nodes)
            )
            print(spacing + line)
            index += level_nodes
            level += 1
        

    def get_parent(self, i:int):
        idx = math.floor((i-1)//2)
        return self.tree[idx]
    
    def get_lchild(self, i:int):
        idx = 2*i
        return self.tree[idx]
    def get_lchild_idx(self, i:int):
        idx = 2*i
        return idx
    
    def get_rchild(self, i:int):
        idx = 2*i + 1
        return self.tree[idx]
    def get_rchild_idx(self, i:int):
        idx = 2*i + 1
        return idx
    
    def is_leaf(self, i:int):
        """
        Returns True if the node at index i is a leaf node
        False otherwise
        """
        # If the idx is greater than the maximum number of nodes, then it's a leaf node
        if i >= 2**(self.md - 1) - 1:
            return True
        # If the left and right children are None, then it's a leaf node
        if self.get_lchild(i) == None and self.get_rchild(i) == None:
            return True
        return False
    
    def get_leaves(self):
        return [i for i in self.tree if self.is_leaf(i)]
    def get_op_nodes(self):
        return [i for i in self.tree if not self.is_leaf(i)]

    def get_depth(self, i:int):
        """
        Returns the depth of the node at index i
        """
        depth = 0
        while i > 0:
            i = math.floor((i-1)//2)
            depth += 1
        return depth
    
    def generate_random_tree_growfull(full):
        # TODO: Implement this
        pass

    def swap_subtrees(tree1, tree2, idx1, idx2):
        """
        Swap the subtrees rooted at idx1 in tree1 and idx2 in tree2.
        """
        def get_subtree(tree, i):
            if i >= len(tree) or tree[i] is None:
                return []
            subtree = []
            queue = [i]
            while queue:
                current = queue.pop(0)
                if current < len(tree):
                    subtree.append(tree[current])
                    queue.append(2 * current + 1)
                    queue.append(2 * current + 2)
                else:
                    subtree.append(None)
            return subtree

        def set_subtree(tree, subtree, i):
            queue = [i]
            idx = 0
            while queue:
                current = queue.pop(0)
                if idx < len(subtree):
                    if current < len(tree):
                        tree[current] = subtree[idx]
                    else:
                        # Extend tree to accommodate the new nodes
                        tree.extend([None] * (current - len(tree) + 1))
                        tree[current] = subtree[idx]
                    queue.append(2 * current + 1)
                    queue.append(2 * current + 2)
                    idx += 1

        # Get the subtrees to be swapped
        subtree1 = get_subtree(tree1, idx1)
        subtree2 = get_subtree(tree2, idx2)
        
        # Replace subtrees in the original trees
        set_subtree(tree1, subtree2, idx1)
        set_subtree(tree2, subtree1, idx2)

        return tree1, tree2


# Example trees
t1 = [1, 2, 3, 4, 5, None, None, 8, 9]
t2 = [10, 20, 30, 40, None, 60]
tree = FastTBTree(4,True)
tree2 = FastTBTree(4,True)

tree.print_tree()
tree2.print_tree()
t1n, t2n = FastTBTree.swap_subtrees(tree.tree, tree2.tree, 1, 0)
t1n = FastTBTree(4,t1n)
t2n = FastTBTree(4,t2n)
t1n.print_tree()
t2n.print_tree()

# Swap subtree rooted at index 1 in t1 with subtree rooted at index 0 in t2
# updated_t1, updated_t2 = swap_subtrees(t1, t2, 1, 0)

# print("Updated Tree 1:", updated_t1)
# print("Updated Tree 2:", updated_t2)

    
    
