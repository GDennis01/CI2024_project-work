from collections import deque
from enum import Enum
import numpy as np
import numpy as np
import warnings
# warnings.filterwarnings("error")
# TODO: finish the remaining mutations. Find a way proper way to select a random node(maybe based on fitness) and apply the mutation to it.

def get_np_functions()->list[tuple[np.ufunc,str,int]]:
    # Get all attributes of numpy module
    all_attrs = dir(np)

    # Filter to include only ufunc objects
    ufuncs = [getattr(np, attr) for attr in all_attrs if isinstance(getattr(np, attr), np.ufunc)]
    # Filter to include only common ufuncs
    ufunc_list = [(ufunc,ufunc.__name__,ufunc.nin) for ufunc in ufuncs if ufunc.__name__ in ['absolute','add','subtract','multiply','divide','sin','cos','sinh','cosh','asin','asinh','acos','acosh','sqrt','exp','exp2','log2','log','maximum','minimum',]]
    return ufunc_list

class NodeType(Enum):
    B_OP=1,
    U_OP=2,
    VAR=3,
    CONST=4

class Node:
    def __init__(self, value, type: NodeType = NodeType.CONST):
        self.value = value
        self.left:Node = None
        self.right:Node = None
        self.type:NodeType = type
    def is_leaf(self):
        return self.left is None and self.right is None
    def evaluate(self,x:np.ndarray):
        """
        Evaluate the tree with the input x
        """
        if self.value is None:
            return None
        if self.type == NodeType.CONST:
            return self.value
        if self.type == NodeType.VAR:
            number = int(self.value[1:])
            return x[number]
        if self.type == NodeType.B_OP:
            left = self.left.evaluate(x)
            right = self.right.evaluate(x)
            try:
                tmp_res = self.value[0](left,right)
            except RuntimeWarning as e:
                print(f'Error in evaluating the node {self.value[1]} with left {left} and right {right}')
            return tmp_res
        if self.type == NodeType.U_OP:
            left = self.left.evaluate(x)
            try:
                tmp_res = self.value[0](left)
            except RuntimeWarning as e:
                print(f'Error in evaluating the node {self.value[1]} with left {left}')

            return tmp_res
    def to_np_formula(self):
        if self.value is None:
            return None
        if self.type == NodeType.CONST:
            return str(self.value)
        if self.type == NodeType.VAR:
            return self.value
        if self.type == NodeType.U_OP:
            left = self.left.to_np_formula()
            return "np."+self.value[1]+"("+left+")"
        if self.type == NodeType.B_OP:
            left = self.left.to_np_formula()
            right = self.right.to_np_formula()
            return "np."+self.value[1]+"("+left+","+right+")"

class BTree:
    def __init__(self):
        self.root:Node = None

    def add_child_at(self, node:Node, value, type:NodeType, position:int):
        """
        Aggiunge una foglia (nuovo node) al node specificato.
        :param node: Node padre a cui aggiungere la foglia.
        :param value: Valore del nuovo node foglia.
        :param position: 0(sinistra) o 1(destra), indica dove aggiungere la foglia.
        """
        new_leaf = Node(value,type)
        if position == 0:
            if node.left is None:
                node.left = new_leaf
            else:
                print("Il node ha già una foglia left.")
        elif position == 1:
            if node.right is None:
                node.right = new_leaf
            else:
                print("Il node ha già una foglia right.")
        else:
            print("Posizione non valida. Usa 0 o 1.")
    def add_child_tail(self, value,type:NodeType):
        """
        Aggiunge un nodo con il value specificato in coda (primo spazio libero).
        :param value: Valore del nodo da aggiungere.
        """
        new_node = Node(value,type)
        if self.root is None:
            # Se l'albero è vuoto, il nuovo nodo diventa la root.
            self.root = new_node
            return

        # Usare una coda per attraversare l'albero in ordine di livello
        queue = deque([self.root])
        while queue:
            current_node = queue.popleft()

            # Verifica lo spazio libero a left
            if current_node.left is None:
                current_node.left = new_node
                return
            else:
                queue.append(current_node.left)

            # Verifica lo spazio libero a right
            if current_node.right is None:
                current_node.right = new_node
                return
            else:
                queue.append(current_node.right)

    def delete_subtree(self, node:Node):
        """
        Elimina un sottoalbero a partire dal node specificato.
        :param node: Node root del sottoalbero da eliminare.
        """
        if node is not None:
            node.value = None
            node.left = None
            node.right = None
        else:
            print("Node non valido. Nessun sottoalbero eliminato.")

    def find_node(self, value)->Node|None:
        """
        Cerca un nodo con un value specifico nell'albero.
        :param value: Valore del nodo da cercare.
        :return: Node trovato o None se non esiste.
        """
        return self._find_node_recursive(self.root, value)
    def get_random_node(self)->Node:
        """
        Get a random node from the tree
        """
        nodes =[self.root]
        self.get_all_nodes(self.root,nodes)
        idx = np.random.randint(0,len(nodes))
        return nodes[idx]
    def get_all_nodes(self, node:Node, nodes):
        if node is not None:
            nodes.append(node)
            self.get_all_nodes(node.left, nodes)
            self.get_all_nodes(node.right, nodes)


    # TODO: fix this
    def _find_node_recursive(self, node:Node, value):
        if node is None:
            return None
        if node.value == value:
            return node
        left = self._find_node_recursive(node.left, value)
        if left:
            return left
        return self._find_node_recursive(node.right, value)

    def get_leaves(self):
        """
        Restituisce una lista di foglie dell'albero.
        """
        leaves = []
        self._get_leaves_recursive(self.root, leaves)
        return leaves
    def _get_leaves_recursive(self, node:Node, leaves):
        if node is not None:
            if node.is_leaf():
                leaves.append(node)
            self._get_leaves_recursive(node.left, leaves)
            self._get_leaves_recursive(node.right, leaves)
    def print_tree(self):
        """
        Stampa l'albero in ordine.
        """
        def _recursive_walk(node, livello=0):
            if node is not None:
                _recursive_walk(node.right, livello + 1)
                print("    " * livello + str(node.value))
                _recursive_walk(node.left, livello + 1)
        _recursive_walk(self.root)

class TweakableBTree(BTree):
    def __init__(self,pm:float,operator_list:list[tuple[np.ufunc,str,int]],n_vars=1):
        """
        pm: probabilità di mutazione
        """
        self.root = BTree()
        self.pm = pm
        self.pr = 1 - pm
        self.operator_list = operator_list
        self.n_vars = n_vars
    def evaluate(self,x:np.ndarray):
        """
        Evaluate the tree with the input x
        """
        return self.root.evaluate(x)
    def to_np_formula(self):
        """
        Convert the tree into a numpy formula
        """
        return self.root.to_np_formula()
    
    
    def gen_random_leaf(n_vars:int)->Node:
        """
        Generate a random leaf node and return it.
        """
        # select between const and vars
        if np.random.rand() < 0.5:
            value = np.random.rand()*100
            new_node = Node(value,NodeType.CONST)
        else:
            value = "x"+str(np.random.randint(0,n_vars))
            new_node = Node(value,NodeType.VAR) 
        return new_node   
    
    def gen_random_op_node(operator_list:list[tuple[np.ufunc,str,int]])->Node:
        # Generate an operator node
            if np.random.rand() < 0.5:
                #Binary function node
                tmp_oplist = list(filter(lambda x: x[2] == 2,operator_list))
                idx = np.random.randint(0,len(tmp_oplist))
                new_node = Node(tmp_oplist[idx],NodeType.B_OP)
            else:
                # Unary function node
                tmp_oplist = list(filter(lambda x: x[2] == 1,operator_list))
                idx = np.random.randint(0,len(tmp_oplist))
                new_node = Node(tmp_oplist[idx],NodeType.U_OP)
            return new_node

    def generate_random_tree_growfull(operator_list:list[tuple[np.ufunc,str,int]],n_vars:int,md:int,full:bool)->Node:
        """
        Generate a random tree with maximum depth md and return it.
        A growfull method is used to generate the tree.
        root: randomly generated root node
        md: maximum depth of the tree
        """
        # Credits: the pseudocode has been taken from this paper https://icog-labs.com/wp-content/uploads/2014/07/Introduction_to_GP_Matthew-Walker.pdf and adapted.

        # if  (not full and np.random.rand() < 0.5 or md == 0) or (full and md == 0):
        if  md == 0 or (not full and np.random.rand() < 0.5):
            new_node = TweakableBTree.gen_random_leaf(n_vars)
            return new_node
        else:
            new_node = TweakableBTree.gen_random_op_node(operator_list)
        new_node.left = TweakableBTree.generate_random_tree_growfull(operator_list,n_vars,md-1,True)
        if new_node.value[2] == 2:
            new_node.right = TweakableBTree.generate_random_tree_growfull(operator_list,n_vars,md-1,True)
        return new_node
        

    def point_mutation(self,node:Node):
        """
        Replace the current node with a new one randomly generated
        """
        match node.type:
            case NodeType.B_OP:
                tmp_oplist = list(filter(lambda x: x[2] == 2,self.operator_list))
                idx = np.random.randint(0,len(tmp_oplist))
                node.value = tmp_oplist[idx]
                pass
            case NodeType.U_OP:
                tmp_oplist = list(filter(lambda x: x[2] == 1,self.operator_list))
                idx = np.random.randint(0,len(tmp_oplist))
                node.value = tmp_oplist[idx]
                pass
            case NodeType.VAR:
                node.value = "x"+str(np.random.randint(0,self.n_vars))
                pass
            case NodeType.CONST:
                node.value = np.random.rand()*100
                pass
    def hoist_mutation(self,node:Node):
        """
        Transform the tree into the selected subtree
        """
        self.root = node


    def collapse_subtree_mutation(self,node:Node):
        """
        Collapse a subtree into a single node
        """
        # 1 - get all children nodes that are not operators
        leaves = self.get_leaves()
        # 2 - select a random node
        idx = np.random.randint(0,len(leaves))
        selected_node = leaves[idx]
        # 3 - replace the current node with the selected node
        node.value = selected_node.value
        node.left = None
        node.right = None
        
    # TODO: expansion and subtree are the same, maybe merge them into a single function
    def expansion_mutation(self,node:Node,md:int):
        """
        Expand a leaf node into a subtree
        """
        assert node.is_leaf()
        # 1 - Generate a new subtree for 
        new_subtree = TweakableBTree.generate_random_tree_growfull(self.operator_list,self.n_vars,md,True)
        # 2 - Replace the current node with the new subtree
        node.left = new_subtree.left
        node.right = new_subtree.right
        node.value = new_subtree.value
        

    def subtree_mutation(self,node:Node,md:int):
        """
        Replace a subtree with another randomly generated subtree
        """
        # 1 - Generate a new subtree
        new_subtree = TweakableBTree.generate_random_tree_growfull(self.operator_list,self.n_vars,md,True)
        # 2 - Replace the current subtree with the new one
        node.left = new_subtree.left
        node.right = new_subtree.right
        node.value = new_subtree.value
        

def main():
    operator_list = get_np_functions()
    n_vars = 2

    # tree = TweakableBTree(0.5,operator_list,1)
    # tree.root = Node(1)
    # node2 = Node(2)
    # node3 = Node(3)
    # tree.root.left = node2
    # tree.root.right = node3

    # # Aggiungere foglie
    # tree.add_child_at(node2, 4,NodeType.CONST, 0)
    # tree.add_child_at(node2, 5,NodeType.CONST, 1)
    # tree.add_child_tail(6,NodeType.CONST,)
    # tree.add_child_tail(6,NodeType.CONST,)
    # tree.add_child_tail(6,NodeType.CONST,)

    # Stampare l'tree
    # print("Albero iniziale:")
    # tree.print_tree()

    # Init a random tree
    tb2 = TweakableBTree(0.5,operator_list,2)
    tb2.root = TweakableBTree.generate_random_tree_growfull(operator_list,n_vars,2,True)

    print("Albero generato:")
    tb2.print_tree()
    # print(tb2.to_np_formula())
    # print(tb2.evaluate(np.array([1,2,3,4,5,6])))

    print("All nodes:")
    nodes = []
    tb2.get_all_nodes(tb2.root,nodes)
    print([node.value for node in nodes])


    print("Albero dopo subtree mutation:")
    node_to_change = tb2.root.left
    tb2.subtree_mutation(node_to_change,3)
    tb2.print_tree()

    print("Albero dopo collapse:")
    tb2.collapse_subtree_mutation(tb2.root.left)
    tb2.print_tree()
    
    # Eliminare un sottotree
    # tree.delete_subtree(node2)

    # Stampare l'tree dopo l'eliminazione
    # print("\nAlbero dopo eliminazione:")
    # tree.print_tree()

    # print("\nMutation:")
    # tree.point_mutation(tree.root)
    # tree.print_tree()


if __name__ == "__main__":
    main()