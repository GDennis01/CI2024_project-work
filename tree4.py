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
    def __str__(self):
        if self.type == NodeType.B_OP or self.type == NodeType.U_OP:
            return str(self.data.__name__)
        return str(self.data)
    def __repr__(self):
        if self.type == NodeType.B_OP or self.type == NodeType.U_OP:
            return str(self.data.__name__)
        return str(self.data)


class FastTBTree:
    def __init__(self, md:int,randomize=False):
        # self.tree:np.ndarray = [None] *2**md - 1
        # self.tree:np.ndarray = np.array([randomize] * (2**md - 1))
        if randomize:
            self.tree = np.random.randint(0, 100, size=(2**md - 1))
        else:
            self.tree = np.array([None] * (2**md - 1))
        self.md = md
    def from_array(md, arr):
        tree = FastTBTree(md)
        tree.tree = arr
        return tree

    def set_params(operators,n_vars,max_const):
        FastTBTree.operators = operators
        FastTBTree.n_vars = n_vars
        FastTBTree.max_const = max_const
        FastTBTree.vars_left = n_vars
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
            spacing = " " * ((max_width // (level_nodes + 1))+1)*2
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
        idx = 2*i+1
        return self.tree[idx]
    def get_lchild_idx( i:int):
        idx = 2*i+1
        return idx
    
    def get_rchild(self, i:int):
        idx = 2*i + 2
        return self.tree[idx]
    def get_rchild_idx( i:int):
        idx = 2*i + 2
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
    
    def gen_random_leaf(n_vars):
        if FastTBTree.vars_left == 0:
            if np.random.rand() < 0.5:
                return Node("x"+str(np.random.randint(0, n_vars)), NodeType.VAR)
            else:
                return Node(np.random.randint(0, FastTBTree.max_const), NodeType.CONST)
        else:
            tmp = Node("x"+str(n_vars - FastTBTree.vars_left), NodeType.VAR)
            FastTBTree.vars_left -= 1
            return tmp
    
    def gen_random_op(operators):
        op = np.random.choice(operators)
        if op.nin == 1:
            return Node(op, NodeType.U_OP)
        return Node(op, NodeType.B_OP)
       

    def generate_random_tree_growfull(full,md):
        tree = np.array([None] * (2**md - 1))
        FastTBTree._generate_random_treegrowfull(tree, 0,md-1,full)
        FastTBTree.vars_left = FastTBTree.n_vars
        return FastTBTree.from_array(md,tree)
        
    def _generate_random_treegrowfull(tree, i,md,full):
        if md == 0 or (not full and np.random.rand() < 0.5):
            new_node = FastTBTree.gen_random_leaf(FastTBTree.n_vars)
            tree[i] = new_node
            return new_node
        new_node = FastTBTree.gen_random_op(FastTBTree.operators)
        tree[i] = new_node
        lindex = FastTBTree.get_lchild_idx(i)
        if lindex < len(tree):
            # tree[lindex] = FastTBTree._generate_random_treegrowfull(tree,lindex,md-1,full)
            FastTBTree._generate_random_treegrowfull(tree,lindex,md-1,full)
        rindex = FastTBTree.get_rchild_idx(i)
        if new_node.type == NodeType.B_OP and rindex < len(tree):
            # tree[rindex] = FastTBTree._generate_random_treegrowfull(tree,rindex,md-1,full)
            FastTBTree._generate_random_treegrowfull(tree,rindex,md-1,full)
        return new_node
    def evalute_tree(self, x):
        return FastTBTree._evalute_tree(self.tree, 0, x)
    def _evalute_tree(tree, i, x):
        node = tree[i]
        if node.type == NodeType.VAR:
            number = int(node.data[1:])
            return x[number]
        if node.type == NodeType.CONST:
            return node.data
        if node.type == NodeType.U_OP:
            return node.data(FastTBTree._evalute_tree(tree, FastTBTree.get_lchild_idx(i), x))
        if node.type == NodeType.B_OP:
            return node.data(FastTBTree._evalute_tree(tree, FastTBTree.get_lchild_idx(i), x), FastTBTree._evalute_tree(tree, FastTBTree.get_rchild_idx(i), x))
    def to_np_formula(self):
        return FastTBTree._to_np_formula(self.tree, 0)
    def _to_np_formula(tree, i):
        node = tree[i]
        if node.type == NodeType.VAR:
            return node.data
        if node.type == NodeType.CONST:
            return str(node.data)
        if node.type == NodeType.U_OP:
            return "np."+node.data.__name__ + "(" + FastTBTree._to_np_formula(tree, FastTBTree.get_lchild_idx(i)) + ")"
        if node.type == NodeType.B_OP:
            return "np."+node.data.__name__ + "(" + FastTBTree._to_np_formula(tree, FastTBTree.get_lchild_idx(i)) + "," + FastTBTree._to_np_formula(tree, FastTBTree.get_rchild_idx(i)) + ")"
        

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
def get_np_functions()->list[tuple[np.ufunc,str,int]]:
    # Get all attributes of numpy module
    all_attrs = dir(np)

    # Filter to include only ufunc objects
    ufuncs = [getattr(np, attr) for attr in all_attrs if isinstance(getattr(np, attr), np.ufunc)]
    # Filter to include only common ufuncs
    ufunc_list = [ufunc for ufunc in ufuncs if ufunc.__name__ in ['absolute','add','subtract','multiply','divide','sin','cos','sinh','cosh','asin','asinh','acos','acosh','sqrt','exp','exp2','log2','log','maximum','minimum',]]
    return ufunc_list
def main():
    # Example generation of random tree
    operator_list = get_np_functions()
    md = 5
    n_vars = 2
    max_const = 100
    FastTBTree.set_params(operator_list,n_vars,max_const)
    tree = FastTBTree.generate_random_tree_growfull(True,md)
    tree.print_tree()
    print(tree.to_np_formula())
    print(tree.tree)
    res = tree.evalute_tree([1,2])
    print(res)

    # Example trees
    # t1 = [1, 2, 3, 4, 5, None, None, 8, 9]
    # t2 = [10, 20, 30, 40, None, 60]
    # tree = FastTBTree(4,True)
    # tree2 = FastTBTree(4,True)

    # tree.print_tree()
    # tree2.print_tree()
    # #TODO: fix this
    # t1n, t2n = FastTBTree.swap_subtrees(tree.tree, tree2.tree, 1, 0)
    # t1n = FastTBTree(4,t1n)
    # t2n = FastTBTree(4,t2n)
    # t1n.print_tree()
    # t2n.print_tree()

    # Swap subtree rooted at index 1 in t1 with subtree rooted at index 0 in t2
    # updated_t1, updated_t2 = swap_subtrees(t1, t2, 1, 0)

    # print("Updated Tree 1:", updated_t1)
    # print("Updated Tree 2:", updated_t2)

if __name__ == "__main__":
    main()
    
    
