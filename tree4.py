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
        self.fitness = None
    def __len__(self):
        return len(self.tree)
    def copy(self):
        return FastTBTree.from_array(self.md,self.tree.copy())
    def from_array(md, arr):
        """
        md: maximum depth of the tree
        arr: np.array of the tree
        """
        tree = FastTBTree(md)
        tree.tree = arr
        return tree

    def set_params(operators,n_vars,max_const):
        FastTBTree.operators = operators
        FastTBTree.n_vars = n_vars
        FastTBTree.max_const = max_const
        FastTBTree.vars_left = n_vars
        FastTBTree.max_leaves = 0 # number of leaves that can be generated at a given instant
    def print_tree(self):
        # check if np array is empty
        if len(self.tree) == 0:
            # print("<empty self.tree>")
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
        if i > len(self.tree):
            return False
        # If the left and right children are None, then it's a leaf node
        if self.get_lchild(i) == None and self.get_rchild(i) == None and self.tree[i] is not None:
            return True
        return False
    def is_op_node(self, i:int):
        if i > len(self.tree):
            return False
        if self.tree[i] is not None and  (self.tree[i].type == NodeType.U_OP or  self.tree[i].type == NodeType.B_OP):
            return True
        return False
    def is_terminal_node(self, i:int):
        if i > len(self.tree):
            return False
        if self.tree[i] is not None and  (self.tree[i].type == NodeType.VAR or  self.tree[i].type == NodeType.CONST):
            return True
        return False
    
    def get_terminal(self):
        # return [self.tree[i] for i in range(len(self.tree)) if self.is_leaf(i)]
        return [self.tree[i] for i in range(len(self.tree)) if self.is_terminal_node(i)]
    def get_terminal_idx(self):
        # return [i for i in range(len(self.tree)) if self.is_leaf(i)]
        return [i for i in range(len(self.tree)) if self.is_terminal_node(i)]

    def get_op_nodes(self):
         return [self.tree[i] for i in range(len(self.tree)) if not self.is_leaf(i)]
    def get_op_nodes_idx(self):
        return [i for i in range(len(self.tree)) if not self.is_leaf(i)]
    def get_depth(self, i:int):
        """
        Returns the depth of the node at index i
        """
        depth = 0
        while i > 0:
            i = math.floor((i-1)//2)
            depth += 1
        return depth
    
    def get_random_terminal(n_vars):
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
        if FastTBTree.max_leaves < FastTBTree.n_vars:
            op = np.random.choice([op for op in operators if op.nin == 2])
            FastTBTree.max_leaves += 1
            return Node(op, NodeType.B_OP)
        op = np.random.choice(operators)
        if op.nin == 1:
            return Node(op, NodeType.U_OP)
        return Node(op, NodeType.B_OP)
       

    def generate_random_tree_growfull(full,md):
        tree = np.array([None] * (2**md - 1))
        FastTBTree._generate_random_treegrowfull(tree, 0,md-1,full)
        FastTBTree.vars_left = FastTBTree.n_vars
        FastTBTree.max_leaves = 0
        t=  FastTBTree.from_array(md,tree)
        t.permutate_leaves()
        return t
        
    def _generate_random_treegrowfull(tree, i,md,full):
        if md == 0 or (not full and np.random.rand() < 0.5):
            new_node = FastTBTree.get_random_terminal(FastTBTree.n_vars)
            tree[i] = new_node
            return new_node
        
        new_node = FastTBTree.gen_random_op(FastTBTree.operators)
        tree[i] = new_node

        lindex = FastTBTree.get_lchild_idx(i)
        if lindex < len(tree):
            FastTBTree._generate_random_treegrowfull(tree,lindex,md-1,full)

        rindex = FastTBTree.get_rchild_idx(i)
        if new_node.type == NodeType.B_OP and rindex < len(tree):
            FastTBTree._generate_random_treegrowfull(tree,rindex,md-1,full)

        return new_node
    def permutate_leaves(self):
        leaves = self.get_terminal()
        leaves_idx = self.get_terminal_idx()
        np.random.shuffle(leaves)
        for i in range(len(leaves_idx)):
            self.tree[leaves_idx[i]] = leaves[i]

        


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
        # # print('Analyzing node:',node)
        if node.type == NodeType.VAR:
            return node.data[0]+"["+node.data[1:]+"]"
        if node.type == NodeType.CONST:
            return str(node.data)
        if node.type == NodeType.U_OP:
            return "np."+node.data.__name__ + "(" + FastTBTree._to_np_formula(tree, FastTBTree.get_lchild_idx(i)) + ")"
        if node.type == NodeType.B_OP:
            return "np."+node.data.__name__ + "(" + FastTBTree._to_np_formula(tree, FastTBTree.get_lchild_idx(i)) + "," + FastTBTree._to_np_formula(tree, FastTBTree.get_rchild_idx(i)) + ")"
        
    def get_subtree(self,i):
        """
        Get the subtree at position i
        i: root of the subtree
        return: FastTBTree subtree at position i
        """
        assert i < len(self.tree)
        if self.is_terminal_node(i):
            return FastTBTree.from_array(self.md,np.array([self.tree[i]]))

        subtree = [None] * len(self.tree)
        st_idx=0
        subtree[st_idx] = self.tree[i]

        queue = [i]
        queue_new_idx = [st_idx]
        while queue:
            current = queue.pop(0)
            st_idx = queue_new_idx.pop(0)
            subtree[st_idx] = self.tree[current]
            lchild = FastTBTree.get_lchild_idx(current)
            if lchild < len(self.tree) and self.tree[lchild] is not None:
                queue_new_idx.append(FastTBTree.get_lchild_idx(st_idx))
                queue.append(lchild)
            rchild = FastTBTree.get_rchild_idx(current)

            if rchild < len(self.tree) and self.tree[rchild] is not None:
                queue_new_idx.append(FastTBTree.get_rchild_idx(st_idx))
                queue.append(rchild)
        return FastTBTree.from_array(self.md,np.array(subtree))
    def set_subtree(self,i,subtree):
        """
        Replace the subtree at position i with a new one
        i: position of the subtree to replace
        subtree: new subtree
        """
        assert i < len(self.tree)
        queue = [i]
        idx = 0
        while queue:
            if idx < len(subtree):
                current = queue.pop(0)
                if current < len(self.tree):
                    self.tree[current] = subtree.tree[idx]
                else:
                    # Extend tree to accommodate the new nodes
                    # self.tree.extend([None] * (current - len(self.tree) + 1))
                    # np.concatenate([self.tree, np.array([None] * (current - len(self.tree) + 1))])
                    # extend the np array
                    self.tree = np.concatenate([self.tree, np.array([None] * (current - len(self.tree) + 1))])
                    self.tree[current] = subtree.tree[idx]
                queue.append(FastTBTree.get_lchild_idx(current))
                queue.append(FastTBTree.get_rchild_idx(current))
                idx += 1
            else:
                break
        
        # set to None the remaining children
        # queue = [queue.pop(0)]
        while queue:
            current = queue.pop(0)
            if current < len(self.tree):
                self.tree[current] = None
                queue.append(FastTBTree.get_lchild_idx(current))
                queue.append(FastTBTree.get_rchild_idx(current))
            else:
                break
        # remove trailing None values
        # find the last non None value
        last_non_none = np.where(self.tree != None)[0][-1]
        self.tree = self.tree[:last_non_none+1]
        return
    def recombination(tree1,tree2):
        """
        Recombine two trees
        tree1: First tree
        tree2: Second tree
        return: New tree
        """
        # TODO: Fix this. If the random subtree is a leaf node, it should be replace to a leaf node
        # If the root of the subtree is a b_op node, then it should be replaced where there is a b_op node in the other tree
        # Same thing for the u_op node

        tree1cpy = tree1.copy()
        tree2cpy = tree2.copy()
        # print('Tree1')
        tree1cpy.print_tree()
        # print(tree1cpy.to_np_formula())
        # print("---")
        # print('Tree2')
        tree2cpy.print_tree()
        # print(tree2cpy.to_np_formula())
        # print("---")
        # Choose a random node from tree1cpy
        non_none_idxs = np.where(tree1cpy.tree != None)[0]
        idx1 = np.random.choice(non_none_idxs)

        # get its subtree
        st1 = tree1cpy.get_subtree(idx1)

        # Choose a random node from tree2cpy
        non_none_idxs = np.where(tree2cpy.tree != None)[0]
        idx2 = np.random.choice(non_none_idxs)

        # get its subtree
        st2 = tree2cpy.get_subtree(idx2)
        # print(f'Subtree 1 at {idx1}:')
        st1.print_tree()
        # print(st1.to_np_formula())
        # print(f'Subtree 2 at {idx2}:')
        st2.print_tree()
        # print(st2.to_np_formula())

        # Swap the subtrees
        # print('Swapping subtrees')

        tree1cpy.set_subtree(idx1,st2)
        # print('tree1cpy with tree2cpy subtree')
        tree1cpy.print_tree()

        tree2cpy.set_subtree(idx2,st1)
        # print('tree2cpy with tree1cpy subtree')
        tree2cpy.print_tree()

        return tree1cpy,tree2cpy
        
def trim_tree_at_depth(tree,md:int):
    """
    Trim the tree at a given depth
    """
    pass



def get_np_functions()->list[np.ufunc]:
    all_attrs = dir(np)

    ufuncs = [getattr(np, attr) for attr in all_attrs if isinstance(getattr(np, attr), np.ufunc)]

    ufunc_list = [ufunc for ufunc in ufuncs if ufunc.__name__ in ['absolute','add','subtract','multiply','divide','sin','cos','sinh','cosh','asin','asinh','acos','acosh','sqrt','exp','exp2','log2','log','maximum','minimum',]]
    return ufunc_list
def main():
    # Example generation of random tree
    np.random.seed(0)
    operator_list = get_np_functions()
    md = 5
    n_vars = 3
    max_const = 100
    FastTBTree.set_params(operator_list,n_vars,max_const)
    
    # print(f'Tree1:')
    tree = FastTBTree.generate_random_tree_growfull(True,md)
    # tree.print_tree()
    # res = tree.evalute_tree([1,2,3,4,5,6,7,9,9,9,9])
    # # print(tree.to_np_formula())

    # print(f'Tree2:')
    tree2 = FastTBTree.generate_random_tree_growfull(True,3)
    # [minimum, x0, maximum, None, None, x0, x0]
    arr = np.array([np.minimum, 'x0', np.maximum, None, None, 'x0', 'x0'])
    treeprova = FastTBTree.from_array(md,np.array(arr))
    treeprova.print_tree()
    print("----------")
    # tree2.print_tree()
    # res = tree2.evalute_tree([1,2,3,4,5,6,7,9,9,9,9])
    # # print(tree2.to_np_formula())
    # # print(tree2.tree)

  
    # # print(f'Tree1 with tree2 at 1:')
    # tree.set_subtree(1,tree2)
    # tree.print_tree()
    # # print(f'Tree2 with tree1 at 3:')
    # tree2.set_subtree(3,tree)
    # tree2.print_tree()
    # # print(tree2.to_np_formula())
    # # print(tree2.evalute_tree([1,2,3,4,5,6,7,9,9,9,9]))
    # print("-------")
    tree,tree2 = FastTBTree.recombination(tree,tree2)
    r1 = tree.evalute_tree([1,2,3,4,5,6,7,9,9,9,9])
    # print(f'Tree1:{tree.to_np_formula()} = {r1}')
    r2 = tree2.evalute_tree([1,2,3,4,5,6,7,9,9,9,9])
    # print(f'Tree2:{tree2.to_np_formula()} = {r2}')
    # tree1.print_tree()
    # tree2.print_tree()
    



if __name__ == "__main__":
    main()
    
    
