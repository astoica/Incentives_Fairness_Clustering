''' In this code we're replacing negative floyd warshall with the SPFA algorithm to find negative cycles;
we use SPFA in two ways: to find negative cycles in G_M and to assert that there are no negative t-cycles in G_new'''
import networkx as nx
import csv 
import numpy as np
import random
import copy
import math 
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pickle
from sklearn.metrics import silhouette_score 

def cluster_proportions(G, list_nodes_G, no_clusters, cluster_assignment):
    # sizes of clusters                                                                                                    
    unique, counts = np.unique(cluster_assignment, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    cluster_proportion = {}
    cluster_majority = {}
    cluster_minority = {}
    for kk in range(no_clusters):
        cluster_proportion[kk] = 0
        cluster_majority[kk] = 0
        cluster_minority[kk] = 0

    for u in range(len(list_nodes_G)):
        if G.nodes[list_nodes_G[u]]['color'] == 'b':
            cluster_proportion[cluster_assignment[u]] += 1
            cluster_majority[cluster_assignment[u]] += 1
        else:
            cluster_minority[cluster_assignment[u]] += 1

    for kk in range(no_clusters):
        cluster_proportion[kk] /= cluster_sizes[kk]
    return cluster_sizes, cluster_proportion, cluster_majority, cluster_minority

def graph_communities(G, list_nodes_G):
    maj_no = 0
    min_no = 0
    for u in range(len(list_nodes_G)):
        if G.nodes[list_nodes_G[u]]['color'] == 'b':
            maj_no += 1
        else:
            min_no += 1
    return maj_no, min_no

def compute_fairness_avgprop(G, list_nodes_G, no_clusters, cluster_assignment):
    fairness_clusters = {}
    [cluster_sizes, cluster_proportion, cluster_majority, cluster_minority] = cluster_proportions(G,list_nodes_G, no_clusters, cluster_assignment)
    [maj_no, min_no] = graph_communities(G, list_nodes_G)
    maj_prop = maj_no / (maj_no + min_no)
    for kk in range(no_clusters):
        fairness_clusters[kk] = abs(cluster_proportion[kk]- maj_prop)
 
    fairness_overall = np.mean(list(fairness_clusters.values()))
    return fairness_clusters, fairness_overall

# this utility function is mfu, from Price of Pareto Optimality in hedonic games by elkind et al, namely, w_i(C) / |C| - 1 (or w_i(C)/|C|) where w_i(C) is the sum of utility of node i in cluster C
def compute_util_avgprop_mfhg(G, list_nodes_G, no_clusters, cluster_assignment):
    util_clusters = {}
    for i in range(no_clusters):
        util_clusters[i] = 0
        # indices in cluster i
        cl = np.where(cluster_assignment == i)[0]
        for j in cl: 
            deg_u = incluster_degree(G, list_nodes_G, cluster_assignment, list_nodes[j])
            util_clusters[i] += deg_u/ (len(cl) - 1)
    util_overall = np.mean(list(util_clusters.values()))
    return util_clusters, util_overall

# this utility function is the closeness utility
def compute_util_avgprop_closeness(G, list_nodes_G, no_clusters, cluster_assignment):
    util_clusters = {}
    for i in range(no_clusters):
        util_clusters[i] = 0
        # indices in cluster i
        cl = np.where(cluster_assignment == i)[0]
        for j in cl: 
            lengths_total = nx.single_source_shortest_path_length(G, list_nodes_G[j])
            deg_u = incluster_degree(G, list_nodes_G, cluster_assignment, list_nodes[j])
            lengths_j = 0
            for jj in cl:
                if j != jj:
                    lengths_j += lengths_total[list_nodes_G[jj]]
            util_clusters[i] += deg_u/lengths_j
    util_overall = np.mean(list(util_clusters.values()))
    return util_clusters, util_overall


def compute_avg_conductance(G, list_nodes_G, no_clusters, cluster_assignment):
    conductance = 0
    conductance_clusters = {}
    
    for i in range(no_clusters):
        ll = np.where(cluster_assignment==i)[0].tolist()
        conductance_clusters[i] = nx.conductance(G, ll, list(set(list_nodes) - set(ll)))
    conductance = np.mean(list(conductance_clusters.values()))
    return conductance_clusters, conductance

def compute_avg_kdistance_cl(cluster_assignment, my_cluster, km_distances):
    avgdist = km_distances[np.where(cluster_assignment==my_cluster)][:,my_cluster].mean()
    return avgdist

def compute_avg_kdistance(cluster_assignment, no_clusters, km_distances):
    dist_clusters = {}
                                                                                                     
    for kkk in range(no_clusters):
        dist_clusters[kkk] = km_distances[np.where(cluster_assignment==kkk)][:,kkk].mean()
    avgdist = np.mean(list(dist_clusters.values()))
    return dist_clusters, avgdist

def compute_avg_ncut(G, list_nodes_G, no_clusters, cluster_assignment):
    ncut = 0
    ncut_clusters = {}
                                                                                                    
    for i in range(no_clusters):
        ll = np.where(cluster_assignment==i)[0].tolist()
        ncut_clusters[i] = nx.normalized_cut_size(G, ll, list(set(list_nodes) - set(ll)))
    ncut = np.mean(list(ncut_clusters.values()))
    return ncut_clusters, ncut

def doubly_weighted_G(G,Gnew, list_nodes_G,cluster_assignment,no_clusters, km_distances):
    [fairness_cl, fairness_all] = compute_fairness_avgprop(G, list_nodes_G, no_clusters, cluster_assignment)
    [distances_cl, distances] = compute_avg_kdistance(cluster_assignment, no_clusters, km_distances)

    for u in G.nodes():
        for v in G.nodes():
            if cluster_assignment[u] != cluster_assignment[v]:
                mycl = cluster_assignment[v]
                cluster_assignment[u], cluster_assignment[v] = cluster_assignment[v], cluster_assignment[u]
                [fairness_cl_new, fairness_all_new] = compute_fairness_avgprop(G, list_nodes_G, no_clusters, cluster_assignment)
                distances_cl_new = compute_avg_kdistance_cl(cluster_assignment, mycl, km_distances)

                Gnew.add_edge(u,v,auv=(fairness_cl_new[mycl]-fairness_cl[mycl]),tuv=distances_cl_new-distances_cl[mycl])
                cluster_assignment[u], cluster_assignment[v] = cluster_assignment[v], cluster_assignment[u]
    # adding edges from nodes to clusters
    cluster_nodes = list(range(len(G.nodes()),len(G.nodes()) + no_clusters))
    Gnew.add_nodes_from(cluster_nodes)
    for u in G.nodes():
        for i in range(no_clusters):
            if cluster_assignment[u] != i:
                ucl = cluster_assignment[u]
                cluster_assignment[u] = i
                v = i + len(G.nodes())
                [fairness_cl_new, fairness_all_new] = compute_fairness_avgprop(G, list_nodes_G, no_clusters, cluster_assignment)
                distances_cl_new = compute_avg_kdistance_cl(cluster_assignment, i, km_distances)               
                Gnew.add_edge(u,v,auv=(fairness_cl_new[i]-fairness_cl[i]),tuv=distances_cl_new-distances_cl[i])
                cluster_assignment[u] = ucl
    # add node start 
    nn = len(Gnew.nodes())
    Gnew.add_node(nn)
    for j in range(len(G.nodes())):
        # add an edge from start node to every node with weight equal to the difference if we remove the node from its cluster
        mycl2 = cluster_assignment[j]
        cluster_assignment[j] = (mycl2 + 1)%no_clusters
        [fairness_cl_new2, fairness_all_new2] = compute_fairness_avgprop(G, list_nodes_G, no_clusters, cluster_assignment)
        distances_cl_new2 = compute_avg_kdistance_cl(cluster_assignment, mycl2, km_distances)

        Gnew.add_edge(nn,j,auv=(fairness_cl_new2[mycl2]-fairness_cl[mycl2]),tuv=distances_cl_new2-distances_cl[mycl2])
        cluster_assignment[j] = mycl2
    for j in range(len(G.nodes()),nn):
        Gnew.add_edge(j,nn,auv=0,tuv=0)
    return Gnew

# find whether there is a negative cycle in t_uv weights
def negCyclefloydWarshall(G): 
    V = len(G.nodes())
    # dist[][] will be the 
    # output matrix that will  
    # finally have the shortest  
    # distances between every 
    # pair of vertices  
    #dist=[[0 for i in range(V+1)]for j in range(V+1)] 
    dist=[[0 for i in range(V)]for j in range(V)] 
       
    # Initialize the solution 
    # matrix same as input 
    # graph matrix. Or we can 
    # say the initial values  
    # of shortest distances 
    # are based on shortest  
    # paths considering no 
    # intermediate vertex.  
    for i in range(V): 
        for j in range(V): 
            if [i,j] in G.edges():
                dist[i][j] = G[i][j]['tuv'] 
            else:
                dist[i][j] = math.inf
    ''' Add all vertices one 
        by one to the set of  
        intermediate vertices. 
    ---> Before start of a iteration, 
         we have shortest 
        distances between all pairs 
        of vertices such  
        that the shortest distances 
        consider only the 
        vertices in set {0, 1, 2, .. k-1} 
        as intermediate vertices. 
    ----> After the end of a iteration, 
          vertex no. k is  
        added to the set of 
        intermediate vertices and  
        the set becomes {0, 1, 2, .. k} '''
    for k in range(V): 
      
        # Pick all vertices  
        # as source one by one 
        for i in range(V): 
                   
            # Pick all vertices as 
            # destination for the 
            # above picked source 
            for j in range(V): 
          
                # If vertex k is on 
                # the shortest path from 
                # i to j, then update 
                # the value of dist[i][j] 
                if ((dist[i][k] + dist[k][j]) < dist[i][j]): 
                        dist[i][j] = dist[i][k] + dist[k][j] 
   
    # If distance of any 
    # vertex from itself 
    # becomes negative, then 
    # there is a negative 
    # weight cycle. 
    for i in range(V): 
        if (dist[i][i] < 0): 
            return True
   
    return False

# create graph with weights a_uv * M - t_uv
def create_M_graph(G,newG, theM):
    myG_M = nx.DiGraph()
    myG_M.add_nodes_from(G)
    auv_var=nx.get_edge_attributes(newG,'auv')
    tuv_var=nx.get_edge_attributes(newG,'tuv')
    for e in newG.edges():
        # note that it is auv + M * tuv, in order to make M positive
        myG_M.add_edge(e[0],e[1],weight=auv_var[e]+theM*tuv_var[e])
    return myG_M

# without early termination and with cycle termination:
def SPFA(G):
    queue = []
    for v in G.nodes():
        length[v] = 0
        dis[v] = 0
        pre[v] = 0
        queue.append(v)
    while len(queue) > 0:
        u = queue.pop(0)
        for (u, v) in G.edges():
            if dis[u] + G[u][v]['weight'] < dis[v]:
                pre[v] = u
                length[v] = length[u] + 1
                if length[v] == len(G.nodes()):
                    return v,pre,"negative cycle detected"
                dis[v] = dis[u] + G[u][v]['weight']
                if v not in queue:
                    queue.append(v)
    return "no negative cycle detected"

# trace the negative cycle from the vertex given by SPFA
def Trace(pre, v):
    mys = []
    while v not in mys:
        mys.append(v)
        v = pre[v]
    cycle = [v]
    while mys[len(mys)-1] != v:
        cycle.append(mys.pop())
    cycle.append(v)
    return cycle

# Slope of line
def compute_slope(x1, y1, x2, y2):
    return (float)(y2-y1)/(x2-x1)

# without early termination and with cycle termination:
def SPFA2(G):
    queue = []
    for v in G.nodes():
        length[v] = 0
        dis[v] = 0
        pre[v] = 0
        queue.append(v)
    while len(queue) > 0:
        u = queue.pop(0)
        for (u, v) in G.edges():
            if dis[u] + G[u][v]['tuv'] < dis[v]:
                pre[v] = u
                length[v] = length[u] + 1
                if length[v] == len(G.nodes()):
                    return v,pre,"negative cycle detected"
                dis[v] = dis[u] + G[u][v]['tuv']
                if v not in queue:
                    queue.append(v)
    return "no negative cycle detected"

# main function to execute
if __name__ == "__main__":
    no_of_iterations = 30
    #iteration = 0
    fairness_list = []
    cost_list = []
    # this code includes the implementation of a stochastic block model; for reading the real data, please see / modify with the code from section5-nashequilibria-realdatasets.py
    # sizes of each block, the length of sizes defines the number of blocks
    sizes = [10, 10, 10, 10, 10]
    # probability of connections 
    probs = [[0.7, 0.07, 0.05, 0.03, 0.05], 
            [0.07, 0.6, 0.1, 0.05, 0.07], 
            [0.05, 0.1, 0.5, 0.05, 0.1], 
            [0.03, 0.05, 0.05, 0.6, 0.05], 
            [0.05, 0.07, 0.1, 0.05, 0.6]]

    # create the graph based on the stochastic block model
    G_SBM = nx.stochastic_block_model(sizes, probs)

    nx.info(G_SBM)
    list_nodes=list(G_SBM.nodes())

    # add a label, red or blue, to each node, in a random fashion; 'ratio' is the ratio of the red nodes
    ratio = 0.3
    no_red = int(0.3*sum(sizes))
    no_blue = sum(sizes) - no_red
    attributes_r = ['r']*no_red
    attributes_b = ['b']*no_blue
    attributes = attributes_r + attributes_b
    random.shuffle(attributes)
    attributes_dict = {}
    count = 0
    for u in G_SBM.nodes():
        attributes_dict[u] = attributes[count]
        count += 1
    nx.set_node_attributes(G_SBM, attributes_dict,'color')

    # k is the number of clusters for spectral clustering
    k = len(sizes)
    # finding the spectrum of the graph
    A = nx.adjacency_matrix(G_SBM)
    L = nx.normalized_laplacian_matrix(G_SBM)
    L.todense()
    D = np.diag(np.sum(np.array(A.todense()), axis=1))

    e, v = np.linalg.eig(L.todense())

    i = [list(e).index(j) for j in sorted(list(e))[1:k]]
    print(i)

 
    U = np.array(v[:, i])
    # performing manual spectral clustering: using the first k values of the eigenspace to do k-means 
    km = KMeans(init='k-means++', n_clusters=k, max_iter=200, n_init=200, verbose=0, random_state=3425)
    km.fit(U)
    y = km.labels_
    # distances from the k-centers
    X_dist = km.transform(U)**2

    # keeping a copy of the clustering assignment
    y_copy = copy.deepcopy(y)
    print(y_copy)

    
    for iteration in range(no_of_iterations):
        print("Iteration number: ",iteration)
        # compute the cost and fairness of this clustering
        [_,avgf] = compute_fairness_avgprop(G_SBM, list_nodes,k,y)
        print("Average unfairness: ", avgf)  
        
        fairness_list.append(avgf)
        [_,avgcost] = compute_avg_kdistance(y,k,X_dist)
        print("Average cost: ", avgcost)
        cost_list.append(avgcost)
        if avgf == 0:
            break
        # generating G_new with double weights
        G_new = nx.DiGraph()
        G_new.add_nodes_from(G_SBM)
        G_new = doubly_weighted_G(G_SBM,G_new,list_nodes,y,k,X_dist)
        list_nodes_Gnew = list(G_new.nodes())

        length = {}
        dis = {}
        pre = {} 
        result = SPFA2(G_new)
        # assert that G_new does not have a negative t-cycle
        if len(result) == 3:
            assert(iteration > 0)

            my_slope = compute_slope(fairness_list[iteration-1], cost_list[iteration-1], fairness_list[iteration], cost_list[iteration])
            myM = -1/my_slope
            epsilon = myM*1e-60
            print("M and slope:", M, myM)
            for e in G_new.edges():
                G_new[e[0]][e[1]]['tuv'] += G_new[e[0]][e[1]]['auv']/(myM+epsilon)
                G_new[e[0]][e[1]]['tuv'] = round(G_new[e[0]][e[1]]['tuv'],10)

        length = {}
        dis = {}
        pre = {} 
        newresult = SPFA2(G_new) 

        if len(newresult) == 3:
            [vv,pre,stri] = newresult
            negt = Trace(pre,vv)
            print(negt)
            sumcycle = 0
            for i in range(len(negt) - 1):
                sumcycle += G_new[negt[i]][negt[i+1]]['tuv']
            print("negative t-cycle was not fixed: ", sumcycle)
            y_neg = np.copy(y)
            for i in range(len(negt)):
                if negt[i] < len(G_SBM.nodes()):
                    if i == len(negt) - 1:
                        break
                    if negt[i+1] < len(G_SBM.nodes()):
                        y_neg[negt[i]] = y[negt[i+1]]
                    if negt[i + 1] >= len(G_SBM.nodes()) and negt[i+1] < len(G_SBM.nodes()) + k:
                        y_neg[negt[i]] = negt[i+1] % len(G_SBM.nodes())
            _,avgdistneg = compute_avg_kdistance(y_neg, k, X_dist)
            _,avgfairnessneg = compute_fairness_avgprop(G_SBM, list_nodes,k,y_neg)
            print("Cost when correcting a negative cycle: ", avgdistneg)
            print("Fairness when correcting a negative cycle: ", avgfairnessneg)

            break

        length = {}
        dis = {}
        pre = {} 

        print("we're finding Mlow")
        
        Mfind = 1
        G_M = create_M_graph(G_SBM,G_new, Mfind)
        mresult = SPFA(G_M)
        if(len(mresult) != 3):
            Mfind = 0
            G_M = create_M_graph(G_SBM,G_new, Mfind)
            assert(len(SPFA(G_M)) == 3)
        else:
            while True:
                G_M = create_M_graph(G_SBM,G_new, Mfind)
                if SPFA(G_M) == 'no negative cycle detected':
                    break
                Mfind *= 2

        print
        print("we found Mlow")
        # initialize M and the limits for the binary search
        M = Mfind/2
        delta = Mfind/2
        low = Mfind/2
        high = Mfind

        # binary search to find M for which there is a cycle of length 0
        termination_condition = 10e-14
        mids = []
        while np.abs(delta) > termination_condition:
            if high > low: 
                mid = (high + low) / 2
                mids.append(mid)
            G_M = create_M_graph(G_SBM,G_new, mid)
            if SPFA(G_M) == 'no negative cycle detected':
                delta = mid - low
                high = mid
            else:
                delta = high - mid
                low = mid
        whereinmidsweare = 0
        for mm in reversed(mids):
            whereinmidsweare += 1
            G_M = create_M_graph(G_SBM,G_new, mm)
            if len(SPFA(G_M)) == 3:
               break
        if whereinmidsweare == len(mids):
            break
        M=mm
        
        G_M = create_M_graph(G_SBM,G_new, M)
        [v,pre,stri]=SPFA(G_M)
        myc = Trace(pre,v)
        print("best cycle: ",myc)
        ytest = np.copy(y)
        for i in range(len(myc)):
            print(myc[i])
            if myc[i] < len(G_SBM.nodes()):
                if i == len(myc) - 1:
                    break
                if myc[i+1] < len(G_SBM.nodes()):
                    #a = y[myc[i]]
                    ytest[myc[i]] = y[myc[i+1]]
                if myc[i + 1] >= len(G_SBM.nodes()) and myc[i+1] < len(G_SBM.nodes()) + k:
                    print(myc[i+1])
                    ytest[myc[i]] = myc[i+1] % len(G_SBM.nodes())
        y = np.copy(ytest)

    print("Fairness: ", fairness_list)
    print("Cost: ", cost_list)
    filename = 'SBM_n' + str(len(list_nodes)) + '_r' + str(ratio) + '_k' + str(k)
    f = open('filename','w')
    writer=csv.writer(f,lineterminator="\n")
    writer.writerow(fairness_list)
    writer.writerow(cost_list)
    f.close()
