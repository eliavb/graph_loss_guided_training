import itertools

from gensim.models import Word2Vec
import numpy as np
import argparse
import networkx as nx

parser = argparse.ArgumentParser(description=('Code used to create the paper: '
                                              '"Self-similar Epochs: Value in arrangement", ICML 2019'))
parser.add_argument('--input_graph', type=str, help='input graph path to embed.')
parser.add_argument('--output_path', type=str, help='output embedding path.')
parser.add_argument('--epochs', type=int, default=10, help='number epochs to train.')
parser.add_argument('--loss_power', type=float, default=1.0, help='Power to raise the loss function.')
parser.add_argument('--walk_prefix', type=int, default=1.0, help='Loss prefix to compute the loss')
parser.add_argument('--walk_length', type=int, default=5.0, help='Random walk length.')
parser.add_argument('--epoch_fraction', type=int, default=5.0, help='fraction of epoch to compute loss.')


def write_embeddings(node_embeddings, output_path):
    node_embeddings.save(output_path)


class RandomWalkSampling:

    def __init__(self, input_graph, model, walk_length, loss_power, walk_prefix, epochs, epoch_fraction):
        self._input_graph = input_graph
        self._model = model
        self._walk_length = walk_length
        self._loss_power = loss_power
        self._walk_prefix = walk_prefix
        self._n = len(input_graph.nodes)
        self._epochs = epochs
        self._epoch_fraction = epoch_fraction

    def get_total_examples(self):
        return self._n * self._epochs

    def _get_random_walk(self, node):
        walk = [node]
        current_node = node
        for _ in range(self._walk_length):
            nbrs = list(self._input_graph.neighbors(current_node))
            next_node = np.random.choice(nbrs)
            walk.append(next_node)
            current_node = next_node
        return walk

    @staticmethod
    def _sigmoid(x):
        return 1 / float(1 + np.exp(x))

    def _get_pair_loss(self, n1, n2):
        return self._sigmoid(np.dot(self._model.wv[n1], self._model.wv[n2]))

    def _get_walk_loss(self, walk):
        loss = 0
        for i in range(self._walk_prefix):
            # Approximation of the full loss, see details in paper.
            loss += self._get_pair_loss(walk[i], walk[i+1])
        return loss ** self._loss_power

    def get_epoch_examples(self):
        for j in range(self._epochs):
            print(f'epoch {j}')
            for i in range(self._epoch_fraction):
                epoch_walks = []
                walk_loss = []
                for node in self._input_graph.nodes:
                    walk = self._get_random_walk(node)
                    epoch_walks.append(walk)
                    walk_loss.append(self._get_walk_loss(walk))
                probs = np.array(walk_loss) / sum(walk_loss)
                sampled_walks_idx = np.random.choice(
                    range(len(epoch_walks)), int(self._n / self._epoch_fraction), p=probs)
                for walk_idx in sampled_walks_idx:
                    yield epoch_walks[walk_idx]


def read_graph(input_path):
    return nx.read_gpickle(input_path)


def split_graph_to_train_and_test(input_graph, percentage_for_test, min_edges_from_sample=20):
    test_edges = []
    size_before_test = len(input_graph.edges)
    test_size = int(size_before_test * percentage_for_test / 100)
    for n in input_graph.nodes:
        nbrs = list(input_graph.neighbors(n))
        if len(nbrs) > min_edges_from_sample:
            np.random.shuffle(nbrs)
            test_edges += [(nbr, n) for nbr in nbrs[:10] if input_graph.degree(nbr) > 1]
        if len(test_edges) > test_size:
            break
    if len(test_edges) > test_size:
        test_edges = test_edges[:test_size]
    input_graph.remove_edges_from(test_edges)
    return test_edges, input_graph


def init_model(input_graph):
    model = Word2Vec(min_count=0, vector_size=10)
    vocab = []
    for (u, v) in input_graph.edges:
        vocab.append([u, v])
    model.build_vocab(vocab)
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    graph = read_graph(args.input_graph)
    model = init_model(graph)
    sampler = RandomWalkSampling(
        graph, model, args.walk_length, args.loss_power, args.walk_prefix, args.epochs, args.epoch_fraction)
    model.train(sampler.get_epoch_examples(), total_examples=sampler.get_total_examples(), epochs=1)
    write_embeddings(model, args.output_path)
