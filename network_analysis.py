# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

data = ["data/higgs-retweet_network.edgelist", 
        "data/higgs-reply_network.edgelist", 
        "data/higgs-mention_network.edgelist",
        "data/higgs-social_network.edgelist"]

titles = ["Higgs Retweet Network", 
          "Higgs Reply Network", 
          "Higgs Mention Network", 
          "Higgs Social Network"]

#----Obtaining the giant connected component----
def load_graph(i:int):
  fh=open(data[i], 'r')
  #graph is directed and weighted
  G = nx.read_weighted_edgelist(fh, create_using=nx.DiGraph)
  giant_cc = max(nx.strongly_connected_components(G), key=len)
  gcc = G.subgraph(giant_cc)
  fh.close()
  return gcc

"""### Reporting Methods

The collection of graphing methods to display attributes of our networks
"""

# def plot_graph_page(g, t):
    # fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    # plots = fig.subplots()
    # fig.suptitle(g_types[t])
    # for xi in range(3):
    # plot_deg_dist_er(g[xi], sfigs[xi, 0], xi)
    # plot_coef_dist(g[xi], sfigs[xi, 1], xi)
    # plot_short_path_dist(g[xi], sfigs[xi, 2], xi)
    # fig.savefig('{}.pdf'.format(g_types[t]))
    #fig.show()


def comm_details(gcc, i: int):
  #---Obtaining communities using the Clauset-Newman-Moore for community detection due to its extreme efficiency and 
  # and because communities are found based on best increase in modularity.
  gcc_communities = list(greedy_modularity_communities(gcc))

  #----Global Clustering Coefficient of Graph G----
  gc_coeff = round(nx.average_clustering(gcc) , 2)

  #----Finding the diameter of giant connected component----
  gcc_diameter = round(nx.diameter(gcc), 2)

  #----Average shortest path length of the giant connected component----
  gcc_avg_path_length = round(nx.average_shortest_path_length(gcc), 2)
  print("Reporting on {}".format(titles[i]))
  print("The giant connected component is a {}".format(gcc))
  print("Communities in the Giant Connected Component: ")
  print(gcc_communities)
  print("The global clustering coefficient of the giant connected component of G is {}".format(gc_coeff))
  print('The average shortest path length of the giant connected component of G is {}'.format(gcc_avg_path_length))
  print('The diameter of the giant connected component of G is {}'.format(gcc_diameter))


#----Node Degree Distribution of Graph G----
def node_degree_plot(g):
  degree_frequency = nx.degree_histogram(g)
  degrees = range(len(degree_frequency))
  plt.figure(figsize=(10, 7))
  plt.loglog(degrees, degree_frequency,'o-')
  plt.xlabel('Degree')
  plt.ylabel('Frequency')
  plt.title('Node Degree Distribution of Graph G')
  plt.show()


# ----Distribution of Local Clustering Coefficient of the Giant Connected Component (GCC) of graph G----
def cluster_coef_plt(g):
  coef = nx.clustering(g)
  g_clust = coef.values()
  plt.figure(figsize=(10, 7))
  plt.hist(g_clust, bins=50, histtype='bar', ec='black')
  plt.xlabel('Clustering')
  plt.ylabel('Frequency')
  plt.title('Distribution of Local Clustering Coefficient of Graph G')
  plt.show()

#----Distribution of the shortest path length of the graph----

# Getting the values from the dict
def func(g):
  lengths = []
  # Looping through the matrix given by the spl algo
  for i in nx.shortest_path_length(g):
    # get the values from the dict
    for len in i[1].values():
      lengths.append(len)
  return collections.Counter(lengths)


def func2(G):
    length = func(G)
    x = list(length.keys())
    y = list(length.values())
    plt.figure(figsize=(10, 7))
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Number of shortest paths')
    plt.title('Distribution of The Shortest Path Lengths')
    xs= np.arange(len(x))
    plt.bar(xs, y)
    plt.xticks(xs, x)
    plt.show()


def plot_short_path_dist(g: nx.Graph):
  x, y = short_path_dist(g)
  xs = np.arange(len(x))
  fig = plt.figure()
  ax = fig.subplots()
  ax.set_title('Shortest Path Lengths')
  ax.set(xlabel='Lengths')
  ax.bar(x, y, align='center')
  plt.xticks(xs, x)
  fig.show()


def short_path_dist(g):
  short = nx.shortest_path_length(gcc)
  q = collections.Counter()
  for s in short:
    q += collections.Counter(s[1].values())
  return q.keys(), q.values()


def graph_report(i: int):
  gcc = load_graph(i)
  comm_details(gcc, i)
  node_degree_plot(gcc)
  cluster_coef_plt(gcc)
  func2(gcc)

"""# Graph Reporting

### Higgs Retweet Network Report

"""
graph_report(0)

"""### Higgs Reply Network Report"""

graph_report(1)

"""### Higgs Mention Network Report"""

graph_report(2)

