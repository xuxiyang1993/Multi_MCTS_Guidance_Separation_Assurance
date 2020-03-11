# from nodes_multi import MultiAircraftNode
# from nodes_secHex import MultiAircraftNode


class MCTS:
    # def __init__(self, node: MultiAircraftNode):
    def __init__(self, node):
        self.root = node

    def best_action(self, simulations, search_depth):
        for _ in range(simulations):
            v = self.tree_policy(search_depth)
            reward = v.rollout(search_depth)
            v.backpropagate(reward)
        return self.root.best_child(c_param=0.)

    def tree_policy(self, search_depth):
        current_node = self.root
        while not current_node.is_terminal_node(search_depth):
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
