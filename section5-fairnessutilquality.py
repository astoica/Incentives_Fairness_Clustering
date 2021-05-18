import networkx as nx
import csv 
import numpy as np
import random
import copy
import math 
import scipy
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pickle
from sklearn.metrics import silhouette_score 

def cut_size(G, S, T=None, weight=None):
    edges = nx.edge_boundary(G, S, T, data=weight, default=1)
    if G.is_directed():
        edges = chain(edges, nx.edge_boundary(G, T, S, data=weight, default=1))
    return sum(weight for u, v, weight in edges)

def incluster_degree(G, list_nodes_G, cluster_assignment, node):
    degu = 0
    for nbr in G.neighbors(node):
        #print(nbr)
        if cluster_assignment[list_nodes_G.index(node)] == cluster_assignment[list_nodes_G.index(nbr)]:
            #print("same cluster")
            degu += 1
    return degu

def utility_node_divided(G, list_nodes_G, cluster_assignment, no_clusters, threshold, node):
    lengths_total = nx.single_source_shortest_path_length(G, node)
    lengths = {}
    for i in range(no_clusters):
        lengths[i] = 0 
        cl = np.where(cluster_assignment == i)[0]
        for j in cl: 
            lengths[i] += lengths_total[list_nodes_G[j]]
    #deg_u = incluster_degree(G, list_nodes_G, cluster_assignment, node)
    cluster_assignment_counterfactual = {}

    utility = {}
    for i in range(no_clusters):
        cluster_assignment_counterfactual[i] = copy.deepcopy(cluster_assignment)
        cluster_assignment_counterfactual[i][list_nodes_G.index(node)] = i
        deg_u = incluster_degree(G, list_nodes_G, cluster_assignment_counterfactual[i], node)
        utility[i] = threshold * deg_u / lengths[i] 
    return utility

# this utility function is from Price of Pareto Optimality in hedonic games by elkind et al, namely, w_i(C) / |C| - 1 (or w_i(C)/|C|) where w_i(C) is the sum of utility of node i in cluster C
def utility_node_elkind(G, list_nodes_G, cluster_assignment, no_clusters, threshold, node):
    #lengths_total = nx.single_source_shortest_path_length(G, node)
    #lengths = {}
    #for i in range(no_clusters):
    #    lengths[i] = 0 
    #    cl = np.where(cluster_assignment == i)[0]
    #    for j in cl: 
    #        lengths[i] += lengths_total[list_nodes_G[j]]
    #deg_u = incluster_degree(G, list_nodes_G, cluster_assignment, node)
    cluster_assignment_counterfactual = {}

    utility = {}
    for i in range(no_clusters):
        cluster_assignment_counterfactual[i] = copy.deepcopy(cluster_assignment)
        cluster_assignment_counterfactual[i][list_nodes_G.index(node)] = i
        deg_u = incluster_degree(G, list_nodes_G, cluster_assignment_counterfactual[i], node)
        utility[i] = threshold * deg_u / len(np.where(cluster_assignment_counterfactual[i] == i)[0]) 
    return utility

# this utility function is from Price of Pareto Optimality in hedonic games by elkind et al, namely, w_i(C) / |C| - 1 (or w_i(C)/|C|) where w_i(C) is the sum of utility of node i in cluster C
def utility_node_elkind1(G, list_nodes_G, cluster_assignment, no_clusters, threshold, node):
    #lengths_total = nx.single_source_shortest_path_length(G, node)
    #lengths = {}
    #for i in range(no_clusters):
    #    lengths[i] = 0 
    #    cl = np.where(cluster_assignment == i)[0]
    #    for j in cl: 
    #        lengths[i] += lengths_total[list_nodes_G[j]]
    #deg_u = incluster_degree(G, list_nodes_G, cluster_assignment, node)
    cluster_assignment_counterfactual = {}

    utility = {}
    for i in range(no_clusters):
        cluster_assignment_counterfactual[i] = copy.deepcopy(cluster_assignment)
        cluster_assignment_counterfactual[i][list_nodes_G.index(node)] = i
        deg_u = incluster_degree(G, list_nodes_G, cluster_assignment_counterfactual[i], node)
        utility[i] = threshold * deg_u / (len(np.where(cluster_assignment_counterfactual[i] == i)[0]) - 1)
    return utility

def compute_conductance(G, list_of_nodes_G, cluster_assignment, no_of_clusters):
    clconductance = {}
    for i in range(no_of_clusters):
        icl = [list_of_nodes_G[x] for x in list(np.where(cluster_assignment == i)[0])]
        clconductance[i] = nx.conductance(G,icl)
    return 1 - sum(clconductance.values())/len(clconductance)

def compute_balance(G,list_of_nodes_G,cluster_assignment,no_of_clusters, sensitive_info):
    balance_avg = 0
    for cl in range(no_of_clusters):
        
        clind = np.where(cluster_assignment == cl)[0]
        sens0cl = 0
        sens1cl = 0
        for j in clind:
            if sensitive_info[j] == 0:
                sens0cl += 1
            else:
                sens1cl +=1
        print(sens0cl,sens1cl)
        if sens0cl > 0 and sens1cl > 0:
            balance_cl = min(sens0cl/sens1cl,sens1cl/sens0cl)
        else:
            balance_cl = 0
        balance_avg += balance_cl
    return balance_avg/no_of_clusters

'''This code implements (translates from matlab) fair clustering, where fairness is defined as statistical parity 
for the sensitive attribute [from https://github.com/matthklein/fair_spectral_clustering/blob/master/Fair_SC_normalized.m]'''

#function clusterLabels = Fair_SC_normalized(adj,k,sensitive)
#implementation of fair normalized SC as stated in Alg. 3 
#
#INPUT:
#adj ... (weighted) adjacency matrix of size n x n
#k ... number of clusters
#sensitive ... vector of length n encoding the sensitive attribute 
#
#OUTPUT:
#clusterLabels ... vector of length n comprising the cluster label for each
#                  data point

def Fair_SC_normalized(G, adj,no_clusters,sensitive):
    # MATLAB: n = size(adj, 1);
    n = np.shape(adj)[1]
    
    #converting sensitive to a vector with entries in [h] and building F %%%
#     sens_unique=unique(sensitive);
    sens_unique = np.unique(sensitive)
    
#     h = length(sens_unique);
    h = len(sens_unique)
#     sens_unique=reshape(sens_unique,[1,h]);
    #sens_unique=np.reshape(sens_unique,[1,h], order="F")
    
#     sensitiveNEW=sensitive;
    sensitiveNEW=sensitive.copy()
    
#     temp=1;
    temp = 0
    
#     for ell=sens_unique
#         sensitiveNEW(sensitive==ell)=temp;
#         temp=temp+1;
#     end
    for ell in sens_unique:
        sensitiveNEW[np.where(sensitive==ell)[0]] = temp
        temp += 1


#     F=zeros(n,h-1);
    F=np.zeros([n,h-1])

#     for ell=1:(h-1)
#         temp=(sensitiveNEW == ell);
#         F(temp,ell)=1; 
#         groupSize = sum(temp);
#         F(:,ell) = F(:,ell)-groupSize/n;
#     end
    for ell in range(h-1):
        temp = np.where(sensitiveNEW == ell)[0]
        F[temp,ell]=1
        groupSize = len(temp)
        F[:,ell] = F[:,ell]-groupSize/n

#     degrees = sum(adj, 1);
#     D = diag(degrees);
#     L = D-adj;
    L = nx.normalized_laplacian_matrix(G)
    L.todense()
    D = np.diag(np.sum(np.array(adj.todense()), axis=1))

#     Z = null(F');
#     Q=sqrtm(Z'*D*Z);
#     Qinv=inv(Q);
    _,Z = null(F.transpose())
    zz = ((Z.transpose()).dot(D)).dot(Z)
    # this needs to be checked!! 
    Q = scipy.linalg.sqrtm(zz)
    Q = Q.real
    Qinv = np.linalg.inv(Q)
    
#     Msymm=Qinv'*Z'*L*Z*Qinv;
    Msymm = ((((Qinv.transpose()).dot(Z.transpose())).dot(L.todense())).dot(Z)).dot(Qinv)
#     Msymm=(Msymm+Msymm')/2;
    Msymm = (Msymm+Msymm.transpose())/2
    print("computed a bunch of matrices up to Msymm")
#     try
#         [Y, eigValues] = eigs(Msymm,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(size(Msymm,1),max(2*k,25)));
#     catch
#         [Y, eigValues] = eigs(Msymm,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(size(Msymm,1),max(2*k,25)));
#     end

    #e,v = np.linalg.eig(Msymm.todense())
    e,v = np.linalg.eig(Msymm)
    print("computed Msymm spectrum")
    
    i = [list(e).index(j) for j in sorted(list(e))[1:no_clusters]]
    #print(i)
    Y = np.array(v[:, i])
    Y = Y.real

#     H = Z*Qinv*Y;
    H = (Z.dot(Qinv)).dot(Y)
    
    print("ready for kmeans on H")
    #clusterLabels = kmeans(H,k,'Replicates',10);
    km_fair = KMeans(init='k-means++', n_clusters=no_clusters, max_iter=200, n_init=200, verbose=0, random_state=3425)
    km_fair.fit(H)
    clusterLabels = km_fair.labels_
    #end
    return clusterLabels

def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

### the following section reads in one of the datasets: APS, Facebook, Highschool; uncomment for the data desired to use
'''#APS dataset:
# this is an earlier, smaller version of DBLP with ~53k nodes
G_og = nx.read_gexf('Downloads/clustering_plotting/APS/sampled_APS_pacs052030.gexf')

gg = sorted(nx.connected_components(G_og),key=len,reverse=True)[0]
Gc = G_og.subgraph(gg)
print(nx.info(Gc))

list_nodes=list(Gc.nodes())
print("read the APS graph")

# k is the number of clusters for spectral clustering
#k = 4
# finding the spectrum of the graph
A = nx.adjacency_matrix(Gc)

L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))

e, v = np.linalg.eig(L.todense())
tau = 20
sensitive = []
for u in list_nodes:
    if (Gc.nodes[u]['pacs'] == '05.30.-d'):
        sensitive.append(1)
    else:
        sensitive.append(0)
sensitive = np.array(sensitive)
dataset = 'APS'
'''

'''#Facebook dataset:
Gc = nx.read_edgelist('Downloads/facebook_combined.txt')

print(nx.info(Gc))

list_nodes=list(Gc.nodes())
print("read the Facebook graph")

# k is the number of clusters for spectral clustering
#k = 4
# finding the spectrum of the graph
A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))

e, v = np.linalg.eig(L.todense())
tau = 1

gender = {}
egos = ['0', '107','348','414','686','698','1684','1912','3437','3980']
genderfeatfinder = {}

for u in egos: 
    print(u)
    genderfeatfinder[u] = {}
    filenamefeat = 'Downloads/facebook/' + u + '.featnames'
    ffeat = open(filenamefeat)
    readerfeat = csv.reader(ffeat)
    for rowfeat in readerfeat:
        myrowfeat = rowfeat[0].split()
        genderfeatfinder[u][myrowfeat[0]] = myrowfeat[1].split(';')[0]
    ffeat.close()
    gender_ind = [k for k,v in genderfeatfinder[u].items() if v == 'gender']
    #print(gender_ind)
    filenameego= 'Downloads/facebook/' + u +'.egofeat'
    fego = open(filenameego)
    readerego =csv.reader(fego)
    for rowego in readerego:
        myrowego = rowego[0].split()
        gender[u] = myrowego[int(max(gender_ind))]
    fego.close()
    filename= 'Downloads/facebook/' + u +'.feat'
    f = open(filename)
    reader =csv.reader(f)
    for row in reader:
        myrow = row[0].split()
        #print(myrow)
        user = myrow[0]
        #print(user)
        #g1 = myrow[78]
        #g2 = myrow[79]
        gender[user] = myrow[int(max(gender_ind))+1]
    f.close()

sensitive = []
for u in list_nodes:
    if (gender[u] == '1'):
        sensitive.append(1)
    else:
        sensitive.append(0)
sensitive = np.array(sensitive)
dataset = 'Facebook'
'''

'''#Highschool dataset:
# this is a dataset from a Highschool netwrol from: http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/
G_og = nx.read_edgelist('Downloads/Friendship-network_data_2013.csv')

gg = sorted(nx.connected_components(G_og),key=len,reverse=True)[0]
Gbig = G_og.subgraph(gg)
Gc = Gbig.copy()
print("read the Highschool graph")

# k is the number of clusters for spectral clustering
#k = 4
# finding the spectrum of the graph

gender = {}

filename = 'Downloads/metadata_2013.txt'
f = open(filename)
reader=csv.reader(f)

for row in reader:
    myrow = row[0].split('\t')
    gender[myrow[0]] = myrow[2]

list_init = list(Gc.nodes())
for u in list_init:
    if gender[u] == 'Unknown':
        Gc.remove_node(u)
        
list_nodes = list(Gc.nodes())
print(nx.info(Gc))

A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))

e, v = np.linalg.eig(L.todense())
tau = 1
sensitive = []
for u in list_nodes:
    if gender[u] == 'F':
        sensitive.append(1)
    else:
        sensitive.append(0)
sensitive = np.array(sensitive)
dataset = 'Highschool'
'''



cluster_iter = [2,3,4,5,6,7,8]
util_divided = {}
util_elkind1 = {}

conductance_SC = {}
conductance_SC['sc'] = {}
conductance_SC['scfair'] = {}

balance_SC = {}
balance_SC['sc'] = {}
balance_SC['scfair'] = {}

util_divided['0'] = {}
util_divided['1'] = {}
util_divided['all'] = {}
util_divided['0']['sc'] = {}
util_divided['1']['sc'] = {}
util_divided['all']['sc'] = {}
util_divided['0']['scfair'] = {}
util_divided['1']['scfair'] = {}
util_divided['all']['scfair'] = {}
util_elkind1['0'] = {}
util_elkind1['1'] = {}
util_elkind1['all'] = {}
util_elkind1['0']['sc'] = {}
util_elkind1['1']['sc'] = {}
util_elkind1['all']['sc'] = {}
util_elkind1['0']['scfair'] = {}
util_elkind1['1']['scfair'] = {}
util_elkind1['all']['scfair'] = {}

for k in cluster_iter:
    print("no of clusters: ", k)
    
    sc = SpectralClustering(n_clusters=k,affinity="precomputed",n_init=200)
    ysc = sc.fit_predict(A)
    y_scfair = Fair_SC_normalized(Gc,A,k,sensitive)
    

    util_divided['0']['sc'][k] = 0
    util_divided['1']['sc'][k] = 0
    util_elkind1['0']['sc'][k] = 0
    util_elkind1['1']['sc'][k] = 0
    util_divided['all']['sc'][k] = 0
    util_elkind1['all']['sc'][k] = 0

    util_divided['0']['scfair'][k] = 0
    util_divided['1']['scfair'][k] = 0
    util_elkind1['0']['scfair'][k] = 0
    util_elkind1['1']['scfair'][k] = 0
    util_divided['all']['scfair'][k] = 0
    util_elkind1['all']['scfair'][k] = 0

    conductance_SC['sc'][k] = 0
    conductance_SC['scfair'][k] = 0
    balance_SC['sc'][k] = 0
    balance_SC['scfair'][k] = 0
    

    conductance_SC['sc'][k] = compute_conductance(Gc, list_nodes, ysc, k)
    conductance_SC['scfair'][k] = compute_conductance(Gc, list_nodes, y_scfair, k)
    balance_SC['sc'][k] = compute_balance(Gc,list_nodes,ysc,k, sensitive)
    balance_SC['scfair'][k] = compute_balance(Gc,list_nodes,y_scfair,k, sensitive)

    for u in list_nodes:
        
        xscd = utility_node_divided(Gc, list_nodes, ysc, k, tau, u)[ysc[list_nodes.index(u)]]
        xscfaird = utility_node_divided(Gc, list_nodes, y_scfair, k, tau, u)[y_scfair[list_nodes.index(u)]]
        xscelk1 = utility_node_elkind1(Gc, list_nodes, ysc, k, tau, u)[ysc[list_nodes.index(u)]]
        xscfairelk1 = utility_node_elkind1(Gc, list_nodes, y_scfair, k, tau, u)[y_scfair[list_nodes.index(u)]]
        
        util_divided['all']['sc'][k] += xscd
        util_elkind1['all']['sc'][k] += xscelk1
        util_divided['all']['scfair'][k] += xscfaird
        util_elkind1['all']['scfair'][k] += xscfairelk1
        
        if gender[u] == '1':
            util_divided['1']['sc'][k] += xscd
            util_elkind1['1']['sc'][k] += xscelk1
            util_divided['1']['scfair'][k] += xscfaird
            util_elkind1['1']['scfair'][k] += xscfairelk1
        else:
            util_divided['0']['sc'][k] += xscd
            util_elkind1['0']['sc'][k] += xscelk1
            util_divided['0']['scfair'][k] += xscfaird
            util_elkind1['0']['scfair'][k] += xscfairelk1

for kk in cluster_iter:
    util_divided['0']['sc'][kk] /= len(np.where(sensitive == 0)[0])
    util_divided['1']['sc'][kk] /= len(np.where(sensitive == 1)[0])
    util_divided['all']['sc'][kk] /= len(sensitive)
    util_divided['0']['scfair'][kk] /= len(np.where(sensitive == 0)[0])
    util_divided['1']['scfair'][kk] /= len(np.where(sensitive == 1)[0])
    util_divided['all']['scfair'][kk] /= len(sensitive)
    util_elkind1['0']['sc'][kk] /= len(np.where(sensitive == 0)[0])
    util_elkind1['1']['sc'][kk] /= len(np.where(sensitive == 1)[0])
    util_elkind1['all']['sc'][kk] /= len(sensitive)
    util_elkind1['0']['scfair'][kk] /= len(np.where(sensitive == 0)[0])
    util_elkind1['1']['scfair'][kk] /= len(np.where(sensitive == 1)[0])
    util_elkind1['all']['scfair'][kk] /= len(sensitive)

conductance_SC_list = sorted(conductance_SC['sc'].items())
conductance_SC_fair_list = sorted(conductance_SC['scfair'].items())
xc,yc = zip(*conductance_SC_list)
xcf,ycf = zip(*conductance_SC_fair_list)
plt.plot(xc,yc,color='k',label="Spectral Clustering")
plt.plot(xcf,ycf,'k--',label="Fair Spectral Clustering")
plt.grid(linestyle = '--')

plt.xlabel("Number of clusters")
plt.ylabel("Conductance")
plt.legend()
filename = dataset + '_conductance_SCFairSC.pdf'
plt.savefig(filename)

listd_all = sorted(util_divided['all']['sc'].items())
xdall, ydall = zip(*listd_all)
listdfair_all = sorted(util_divided['all']['scfair'].items())
xdfall, ydfall = zip(*listdfair_all)
list0d = sorted(util_divided['0']['sc'].items())
x0d, y0d = zip(*list0d)
list1d = sorted(util_divided['1']['sc'].items())
x1d, y1d = zip(*list1d)
list0df = sorted(util_divided['0']['scfair'].items())
x0df, y0df = zip(*list0df)
list1df = sorted(util_divided['1']['scfair'].items())
x1df, y1df = zip(*list1df)

plt.plot(xdall,ydall,color='k',label='All SC')
plt.plot(xdfall,ydfall,'k--',label='All Fair SC')
plt.plot(x0d,y0d,color='r',label='Minority SC')
plt.plot(x1d,y1d,color='b',label='Majority SC')
plt.plot(x0df,y0df,'r--',label='Minority Fair SC')
plt.plot(x1df,y1df,'b--',label='Majority Fair SC')
plt.grid(linestyle = '--')

plt.xlabel('Number of clusters')
plt.ylabel('Average utility per node')
plt.legend()
filename = dataset + '_utilSCvsFairSC_dividedutil.pdf'
plt.savefig(filename)

liste1_all = sorted(util_elkind1['all']['sc'].items())
xe1all, ye1all = zip(*liste1_all)
liste1f_all = sorted(util_elkind1['all']['scfair'].items())
xe1fall, ye1fall = zip(*liste1f_all)

list0e1 = sorted(util_elkind1['0']['sc'].items())
x0e1, y0e1 = zip(*list0e1)
list1e1 = sorted(util_elkind1['1']['sc'].items())
x1e1, y1e1 = zip(*list1e1)
list0e1f = sorted(util_elkind1['0']['scfair'].items())
x0e1f, y0e1f = zip(*list0e1f)
list1e1f = sorted(util_elkind1['1']['scfair'].items())
x1e1f, y1e1f = zip(*list1e1f)

plt.plot(xe1all,ye1all,color='k',label='All SC')
plt.plot(xe1fall,ye1fall,'k--',label='All Fair SC')
plt.plot(x0e1,y0e1,color='r',label='Minority SC')
plt.plot(x1e1,y1e1,color='b',label='Majority SC')
plt.plot(x0e1f,y0e1f,'r--',label='Minority Fair SC')
plt.plot(x1e1f,y1e1f,'b--',label='Majority Fair SC')
plt.grid(linestyle = '--')

plt.xlabel('Number of clusters')
plt.ylabel('Average utility per node')
plt.legend()
filename = dataset + '_utilSCvsFairSC_elkind1util.pdf'
plt.savefig(filename)

balance = min(len(np.where(sensitive == 0)[0]) / len(np.where(sensitive == 1)[0]),len(np.where(sensitive == 1)[0]) / len(np.where(sensitive == 0)[0]))
balance_SC_list = sorted(balance_SC['sc'].items())
balance_SC_fair_list = sorted(balance_SC['scfair'].items())
xb,yb = zip(*balance_SC_list)
xbf,ybf = zip(*balance_SC_fair_list)
plt.plot(xb,yb,color='k',label="Spectral Clustering")
plt.plot(xbf,ybf,'k--',label="Fair Spectral Clustering")
plt.grid(linestyle = '--')

plt.hlines(balance,cluster_iter[0],cluster_iter[-1],linestyle='-.', color='green',label='APS Balance')
plt.xlabel("Number of clusters")
plt.ylabel("Average balance")
plt.legend()
filename = dataset + '_avgbalanceSCFairSC.pdf'
plt.savefig(filename)