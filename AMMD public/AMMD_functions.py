import networkx as nx
from networkx.algorithms.flow import dinitz
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import deque
from itertools import chain
from itertools import combinations
import tsplib95
import bisect
import time
from math import inf
import numpy as np

###### Non-TSP data #####

c_elegans = "./datasets/weighted_digraphs/celegansneural_weighted.txt"
moreno_health = "./datasets/weighted_digraphs/moreno_health_weighted.txt"
wiki_vote_snap = "./datasets/weighted_digraphs/Wiki-Vote.txt"
gnutella_snap = "./datasets/weighted_digraphs/Gnutella_snap.txt"
bitcoin = './datasets/weighted_digraphs/bitcoin.txt'
airport = './datasets/weighted_digraphs/airport.txt'
openflight = './datasets/weighted_digraphs/openflight.txt'
cora = './datasets/weighted_digraphs/cora.txt'

ft70 = './datasets/ALL_atsp/ft70.atsp'
kro124 = './datasets/ALL_atsp/kro124p.atsp'
rbg323 = './datasets/ALL_atsp/rbg323.atsp'


##########################################################################################
############### Experiments ##############################################################
##########################################################################################

### Create synthetic instances ###

def create_scc_scale_free(nodesize):
    G = nx.scale_free_graph(nodesize)
    largest_cc = max(nx.weakly_connected_components(G), key=len) # largest weakly connected component.
    G.remove_nodes_from([n for n in G.nodes() if n not in largest_cc]) # removing other nodes.
    
    Grev = G.reverse(copy=True)
    G = nx.compose(G,Grev)
    G = nx.DiGraph(G)

    # Give the edges some random weights if needed.
    #for (u, v) in G.edges():
    #    G[u][v]['weight'] = np.random.randint(1,50)
    
    p = dict(nx.shortest_path_length(G, weight="weight"))
    p2 = dict()
    for u, v in p.items():
        vnew = {x: {"weight": y} for x,y in v.items()}
        p2[u] = vnew
    D = nx.DiGraph(p2)
    D.remove_edges_from(nx.selfloop_edges(D))
    D_weights = nx.get_edge_attributes(D,'weight')

    return D, D_weights

def create_complete_digraph(nodesize, max_range_weights):
    G = nx.complete_graph(nodesize, nx.DiGraph())
    # Give the edges some random weights if needed.
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(1, max_range_weights)

    p = dict(nx.shortest_path_length(G, weight="weight"))
    p2 = dict()
    for u, v in p.items():
        vnew = {x: {"weight": y} for x,y in v.items()}
        p2[u] = vnew
    D = nx.DiGraph(p2)
    D.remove_edges_from(nx.selfloop_edges(D))
    D_weights = nx.get_edge_attributes(D,'weight')

    return D, D_weights

### Timing experiments ###

def Runningtimes_fixed_k(nodesizes, fixed_k):
    # We check the running time vs graph size, for fixed k.
    # Generate strongly connected scale free graphs.
    # nodesizes is a list of sizes to be checked.
    # We check [Approx1, Approx2, Approx3] algorithms.
    # standaard nodesizes = [50,100,200,400,800,1600,3200,6400]
    n = len(nodesizes)
    results_matrix = np.zeros((n,3))
    i = 0
    while i < n:
        print(i)
        D, D_weights = create_scc_scale_free(nodesizes[i])
        start = time.time()
        x = approx1(D, D_weights, fixed_k)
        end = time.time()
        results_matrix[i,0] = round(end-start,1)

        start = time.time()
        x = approx2(D, D_weights, fixed_k)
        end = time.time()
        results_matrix[i,1] = round(end-start,1)

        start = time.time()
        x = approx3(D, D_weights, fixed_k)
        end = time.time()        
        results_matrix[i,2] = round(end-start,1)
        i += 1

    np.savetxt('2Running_times_fixed_k.txt', results_matrix,fmt='%.1f')
    return results_matrix

def Runningtimes_uniquedistances(max_weight_ranges, n, fixed_k):
    # We check the running time vs the unique distances in the graph, for fixed graph size n.
    # standaard n = 400, max_weight_ranges = [50,100,200,400,800,1600,3200].
    l = len(max_weight_ranges)
    results_matrix = np.zeros((l,3))
    i = 0
    while i < l:
        print(i)
        D, D_weights = create_complete_digraph(n, max_weight_ranges[i])
        start = time.time()
        x = approx1(D, D_weights, fixed_k)
        end = time.time()
        results_matrix[i,0] = round(end-start,1)

        start = time.time()
        x = approx2(D, D_weights, fixed_k)
        end = time.time()
        results_matrix[i,1] = round(end-start,1)

        start = time.time()
        x = approx3(D, D_weights, fixed_k)
        end = time.time()        
        results_matrix[i,2] = round(end-start,1)
        i += 1

    np.savetxt('Running_times_uniquedistances.txt', results_matrix ,fmt='%.1f')
    return results_matrix
    
### Performance experiments ###

def Performance_one_Dataset(file_path, krange):
    # file_paths is a path to the dataset, it assumes *forward* slashes '.../.../'.
    # Tested algorithms = [Approx1, Approx2, Approx3, optimum_MaxIndSet_binsearch, randomk, largest_dmin_next].
    # krange is the range of k values for which we test, e.g. krange = [2, 4, 8, 16, 32, 64].
    # This automatically write to the corrext filename 'Performance_datasetname.txt'
    
    ks = len(krange)
    results_matrix = np.zeros((ks,6)) # Matrix rows are k values, Matrix columns are the different Algorithms.

    if file_path[-4:] == '.txt':
        # Then we have a weighted text file
        D, D_weights = readFile_weighted_lscc(file_path)
    else:
        # Otherwise ATSP file.
        D, D_weights = readFile_tsp(file_path)
    i = 0
    while i < len(krange):
        print(i)
        results_matrix[i,0] = approx1(D, D_weights, krange[i])
        print('approx1 done!')
        results_matrix[i,1] = approx2(D, D_weights, krange[i])
        print('approx2 done!')
        results_matrix[i,2] = approx3(D, D_weights, krange[i])
        print('approx3 done!')
        results_matrix[i,3] = optimum_MaxIndSet_binsearch(D, D_weights, krange[i])
        print('opt done!')
        results_matrix[i,4] = randomk(D, D_weights, krange[i], 10)
        print('randomk!')
        results_matrix[i,5] = largest_dmin_next(D, D_weights, krange[i])
        print('largest dmin next done')
        i += 1
    # Add the krange vector as the first column:
    results_matrix = np.insert(results_matrix, 0, krange, axis=1)

    # Automatically writing to correct file name:
    x = file_path.rsplit('/', 1)[-1]
    name = x.split('.')[0]
    np.savetxt('Performance_'+name+'.txt', results_matrix ,fmt='%.0f')
    return results_matrix

##########################################################################################
############### Data File Reading ########################################################
##########################################################################################

def readFile_weighted_lscc(file_path):
    # Reads a 3 column edge list as a weighted DiGraph, takes the largest strongly connected comp (LSCC).
    # Returns D, the metric closure of the LSCC, as a complete nx.DiGraph.
    # Also returns D_weights, which is simply a dict containing edge weights of D.
    G = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph)
    largest = max(nx.strongly_connected_components(G), key=len) # nodes from largest SCC
    G.remove_nodes_from([n for n in G.nodes() if n not in largest]) # removing other nodes.
    p = dict(nx.shortest_path_length(G, weight="weight")) # metric closure = all-pair shortest paths length (if exists a path!).

    # p is a dict of dicts, we slightly need to redefine in order to read a networkx graph.
    p2 = dict()
    for u, v in p.items():
        vnew = {x: {"weight": y} for x,y in v.items()}
        p2[u] = vnew
    D = nx.DiGraph(p2)
    D.remove_edges_from(nx.selfloop_edges(D))
    D_weights = nx.get_edge_attributes(D,'weight')
    
    return D, D_weights

def readFile_weighted(file_path):
    # Reads a 3 column edge list as a weighted DiGraph.
    # Returns D, the metric closure of the LSCC, as a complete nx.DiGraph.
    # Also returns D_weights, which is simply a dict containing edge weights of D.
    # Since we do not take the largest SCC, some node pair distances might be infinite!
    # We add an 'inf' value for the corresponding edges in D.
    
    G = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph)
    p = dict(nx.shortest_path_length(G, weight="weight")) # metric closure = all-pair shortest paths length.

    # p is a dict of dicts, we slightly need to redefine in order to read a networkx graph.
    p2 = dict()
    for u, v in p.items():
        vnew = {x: {"weight": y} for x,y in v.items()}
        p2[u] = vnew
    D = nx.DiGraph(p2)
    D.remove_edges_from(nx.selfloop_edges(D))

    # Add infinity values for edges that are not present in D!
    D_complement = nx.complement(D)
    nx.set_edge_attributes(D_complement, values = inf, name = 'weight')
    D = nx.compose(D,D_complement)  
    D_weights = nx.get_edge_attributes(D,'weight')


    return D, D_weights

def readFile_tsp(file_path):
    # This reads TSP data from TSPLIB
    # All those graphs are already complete, but we still need to take metric closure to ensure the triangle inequality.
    
    with open(file_path) as f:
        text = f.read()

    problem = tsplib95.parse(text)
    G = problem.get_graph() # creates a NetworkX graph.
    p = dict(nx.shortest_path_length(G, weight="weight"))
    p2 = dict()
    for u, v in p.items():
        vnew = {x: {"weight": y} for x,y in v.items()}
        p2[u] = vnew

    D = nx.DiGraph(p2)
    D.remove_edges_from(nx.selfloop_edges(D))
    D_weights = nx.get_edge_attributes(D,'weight')

    return D, D_weights


##########################################################################################
################ Main Algorithms #########################################################
##########################################################################################

def approx1(D, D_weights, k):
    # D is a NetworkX Digraph, with edge attribute 'weights' the directed distances (satisfying triangle ineq.)
    # k is an integer between 1 and n, denoting the size of the set of vertices we want to return.
    # Vanilla version of our algorithm. We iterate over all possible R values, and draw G_aux if d_ij < R/6k.

    if k<=1 or k>len(D.nodes()):
        raise ValueError("integer k needs to be 2<=k<=n.")

    # A list of the unique distances
    unique_d = list(set([v for v in D_weights.values()]))
    sols = {} # For each R in unique_d, this gives a solution set. If R is in sols, then sols[R] is guaranteed to be consisting of k vertices.
    sols_val = {} # The correpsonding solution value.

    for R in unique_d:
        #print(R)
        D_new = cluster_dmax(D, D_weights, R)
        G_aux = create_auxiliary_graph(R, D_new, D_weights, k) # node ID's still the same as in D.

        # Step 1: Find SCC.
        Gc = nx.condensation(G_aux, scc=None) # reindexed DAG with nodes = (0,1,2,...).
        member_dict = nx.get_node_attributes(Gc, "members") # scc membership dict, {0:{a,b,c}, 1:{d,e},...}.
        check_cycle_exists = [x for x, v in member_dict.items() if len(v)>1]
        if check_cycle_exists:
            # If one component has more than 1 vertex, then there exists a cycle, which must be of length >= 2k+2.
            # print('Cycle found!')
            cyc = find_Chordless_cycle(G_aux)
            sols[R] = cyc[:2*k:2]
            sols_val[R] = div_score(D_weights, sols[R])
        else:
            # If no cycle exists, continue working on the DAG.
            topo_order = list(nx.topological_sort(Gc))

            # Find Max. AntiChain 
            cands = maxAntiChain(Gc, topo_order)
            if len(cands) >= k:
                #print('Large antichain')
                c = cands[:k]
                cands_orig = [list(member_dict[x])[0] for x in c] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                sols[R] = cands_orig
                sols_val[R] = div_score(D_weights, sols[R])
            else:
                # Now we compute the longest shortest path.
                path = longest_shortest_path2(Gc, topo_order)
                #print('Length Path found = '+str(len(path)))
                if len(path) >= 2*k+1:
                    cands = path[:2*k:2]
                    cands_orig = [list(member_dict[x])[0] for x in cands] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                    sols[R] = cands_orig
                    sols_val[R] = div_score(D_weights, sols[R])

    Rmax = max(sols_val.items(), key = lambda x: x[1])[0] # The R-value with largest solution value.
    
    return sols_val[Rmax]

def approx2(D, D_weights, k):
    # D is a NetworkX Digraph, with edge attribute 'weights' the directed distances (satisfying triangle ineq.)
    # k is an integer between 1 and n, denoting the size of the set of vertices we want to return.
    # Second variant of our algorithm.
    # Iterates over all possible R values, but binary searches for a *bigger* treshold value than R/6k, when creating G_aux as e in E <-> d_ij < R/6k.
    # We will look for values in between [R/6k, R/3], *if* we find a solution for R/6k.

    if k<=1 or k>len(D.nodes()):
        raise ValueError("integer k needs to be 2<=k<=n.")

    # A list of the unique distances
    unique_d = list(set([v for v in D_weights.values()]))
    unique_d.sort() # we sort the unique R values from small to largest.
    
    sols = {} # For each R in unique_d, this gives a solution set. If R is in sols, then sols[R] is guaranteed to be consisting of k vertices.
    sols_val = {} # The correpsonding solution value.

    for R in unique_d:
        #print(R)
        D_new = cluster_dmax(D, D_weights, R)
        weight_dict = nx.get_edge_attributes(D_new,'weight')
        G_aux = create_auxiliary_graph(R, D_new, D_weights, k) # node ID's still the same as in D.

        # Step 1: Find SCC.
        Gc = nx.condensation(G_aux, scc=None) # reindexed DAG with nodes = (0,1,2,...).
        member_dict = nx.get_node_attributes(Gc, "members") # scc membership dict, {0:{a,b,c}, 1:{d,e},...}.
        check_cycle_exists = [x for x, v in member_dict.items() if len(v)>1]
        if check_cycle_exists:
            # If one component has more than 1 vertex, then there exists a cycle, which must be of length >= 2k+2.
            # print('Cycle found!')
            cyc = find_Chordless_cycle(G_aux)
            sols[R] = cyc[:2*k:2]
            sols_val[R] = div_score(D_weights, sols[R])
        else:
            # If no cycle exists, continue working on the DAG.
            topo_order = list(nx.topological_sort(Gc))

            # Find Max. AntiChain 
            cands = maxAntiChain(Gc, topo_order)
            if len(cands) >= k:
                #print('Large antichain')
                c = cands[:k]
                cands_orig = [list(member_dict[x])[0] for x in c] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                sols[R] = cands_orig
                sols_val[R] = div_score(D_weights, sols[R])
            else:
                # Now we compute the longest shortest path.
                path = longest_shortest_path2(Gc, topo_order)
                #print('Length Path found = '+str(len(path)))
                if len(path) >= 2*k+1:
                    cands = path[:2*k:2]
                    cands_orig = [list(member_dict[x])[0] for x in cands] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                    sols[R] = cands_orig
                    sols_val[R] = div_score(D_weights, sols[R])


        if R in sols:
            # Now we will try to improve the treshold value, worst-case we just use R/6k again.
            # Interval of distances between [R/6k,R/3]:
            v1 = R/6/k
            if v1 in unique_d:
                a = unique_d.index(v1)
            else:
                a = bisect.bisect(unique_d, v1) # the index of first value larger than v1.

            v2 = R
            if v2 in unique_d:
                b = unique_d.index(v2)
            else:
                b = bisect.bisect(unique_d, v2)

            while a<=b:
                cont_right = False # This checks whether we go right or left in binary search.
                midpoint = (a + b)//2
                R2 = unique_d[midpoint]
                
                G_aux = nx.DiGraph()
                G_aux.add_nodes_from(D_new.nodes())
                edges_filtered = [e for e in D_new.edges() if D_weights[e] < R2]
                G_aux.add_edges_from(edges_filtered)

                # Step 1: Find SCC.
                Gc = nx.condensation(G_aux, scc=None) # reindexed DAG with nodes = (0,1,2,...).
                member_dict = nx.get_node_attributes(Gc, "members") # scc membership dict, {0:{a,b,c}, 1:{d,e},...}.
                check_cycle_exists = [x for x, v in member_dict.items() if len(v)>1]
                if check_cycle_exists:
                    # If one component has more than 1 vertex, then there exists a cycle, which must be of length >= 2k+2.
                    # print('Cycle found!')
                    cyc = find_Chordless_cycle(G_aux)
                    if len(cyc)>2*k:
                        cont_right = True
                        sols[R] = cyc[:2*k:2]
                        sols_val[R] = div_score(D_weights, sols[R])
                else:
                    # If no cycle exists, continue working on the DAG.
                    topo_order = list(nx.topological_sort(Gc))

                    # Find Max. AntiChain 
                    cands = maxAntiChain(Gc, topo_order)
                    if len(cands) >= k:
                        cont_right = True
                        c = cands[:k]
                        cands_orig = [list(member_dict[x])[0] for x in c] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                        sols[R] = cands_orig
                        sols_val[R] = div_score(D_weights, sols[R])
                    else:
                        # Now we compute the longest shortest path.
                        path = longest_shortest_path2(Gc, topo_order)
                        #print('Length Path found = '+str(len(path)))
                        if len(path) >= 2*k+1:
                            cont_right = True
                            cands = path[:2*k:2]
                            cands_orig = [list(member_dict[x])[0] for x in cands] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                            sols[R] = cands_orig
                            sols_val[R] = div_score(D_weights, sols[R])

                if cont_right:
                    a = midpoint+1
                else:
                    b = midpoint-1


    Rmax = max(sols_val.items(), key = lambda x: x[1])[0] # The R-value with largest solution value.
    
    return sols_val[Rmax]




def approx3(D, D_weights, k):
    # D is a NetworkX Digraph, with edge attribute 'weights' the directed distances (satisfying triangle ineq.)
    # k is an integer between 1 and n, denoting the size of the set of vertices we want to return.
    # Vanilla version of our algorithm. We iterate over all possible R values, and draw G_aux if d_ij < R/6k.

    if k<=1 or k>len(D.nodes()):
        raise ValueError("integer k needs to be 2<=k<=n.")

    # A list of the unique distances
    unique_d = list(set([v for v in D_weights.values()]))
    unique_d.sort() # we sort the unique R values from small to largest.

    a = 0
    b = len(unique_d)-1
    
    sols = {} 
    sols_val = {}

    while a<b:
        if b-a==1:
            midpoint = b
        else:
            midpoint = (a + b)//2
        R = unique_d[midpoint]
        #print(R)
        D_new = cluster_dmax(D, D_weights, R)
        G_aux = create_auxiliary_graph(R, D_new, D_weights, k) # node ID's still the same as in D.

        # Find the largest R that still gives a solution of size >=k 
        Gc = nx.condensation(G_aux, scc=None) # reindexed DAG with nodes = (0,1,2,...).
        member_dict = nx.get_node_attributes(Gc, "members") # scc membership dict, {0:{a,b,c}, 1:{d,e},...}.
        check_cycle_exists = [x for x, v in member_dict.items() if len(v)>1]
        topo_order = list(nx.topological_sort(Gc))
        if check_cycle_exists:
            a = midpoint
        elif len(maxAntiChain(Gc, topo_order))>= k or len(longest_shortest_path2(Gc, topo_order)) >= 2*k+1:
            a = midpoint
        else:
            b = midpoint-1
    midpoint = (a + b)//2
    R = unique_d[midpoint]
    #print(R)
    D_new = cluster_dmax(D, D_weights, R)
    #print([v for v in D_new.nodes()])
    # weight_dict = nx.get_edge_attributes(D_new,'weight')
    # This R value now satisfies R >= R^*.
    # Now binary search for a better cutoff.
    v1 = R/6/k
    if v1 in unique_d:
        a = unique_d.index(v1)
    else:
        a = bisect.bisect(unique_d, v1) # the index of first value larger than v1.

    v2 = R
    if v2 in unique_d:
        b = unique_d.index(v2)
    else:
        b = bisect.bisect(unique_d, v2)

    while a<=b:
        cont_right = False # This checks whether we go right or left in binary search.
        midpoint = (a + b)//2
        R2 = unique_d[midpoint]
        G_aux = nx.DiGraph()
        G_aux.add_nodes_from(D_new.nodes())
        edges_filtered = [e for e in D_new.edges() if D_weights[e] < R2]
        G_aux.add_edges_from(edges_filtered)

        # Step 1: Find SCC.
        Gc = nx.condensation(G_aux, scc=None) # reindexed DAG with nodes = (0,1,2,...).
        member_dict = nx.get_node_attributes(Gc, "members") # scc membership dict, {0:{a,b,c}, 1:{d,e},...}.
        check_cycle_exists = [x for x, v in member_dict.items() if len(v)>1]
        if check_cycle_exists:
            # If one component has more than 1 vertex, then there exists a cycle, which must be of length >= 2k+2.
            # print('Cycle found!')
            cyc = find_Chordless_cycle(G_aux)
            if len(cyc)>=2*k:
                cont_right = True
                sols[R2] = cyc[:2*k:2]
                sols_val[R2] = div_score(D_weights, sols[R2])
        else:
            # If no cycle exists, continue working on the DAG.
            topo_order = list(nx.topological_sort(Gc))

            # Find Max. AntiChain 
            cands = maxAntiChain(Gc, topo_order)
            if len(cands) >= k:
                cont_right = True
                c = cands[:k]
                cands_orig = [list(member_dict[x])[0] for x in c] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                sols[R2] = cands_orig
                sols_val[R2] = div_score(D_weights, sols[R2])
            else:
                # Now we compute the longest shortest path.
                path = longest_shortest_path2(Gc, topo_order)
                #print('Length Path found = '+str(len(path)))
                if len(path) >= 2*k+1:
                    cont_right = True
                    cands = path[:2*k:2]
                    cands_orig = [list(member_dict[x])[0] for x in cands] # mapping back to original node ID's. This is 1-to-1 since G_aux is a DAG in this case.
                    sols[R2] = cands_orig
                    sols_val[R2] = div_score(D_weights, sols[R2])

        if cont_right:
            a = midpoint+1
        else:
            b = midpoint-1


    Rmax = max(sols_val.items(), key = lambda x: x[1])[0] # The R-value with largest solution value.
    #print(Rmax)
    return sols_val[Rmax], sols[Rmax]


##########################################################################################
################ Heuristics #########################################################
##########################################################################################

def randomk(D, D_weights, k, repeats):
    # This algorithm picks a random k-subset as solution.
    # Picks the best result found among several repeats
    teller = 1
    best_val = 0
    l = list(D.nodes())
    while teller <= repeats:
        teller += 1
        cands = list(np.random.choice(l, k, replace=False))
        cands_val = div_score(D_weights,cands)
        if cands_val >= best_val:
            best_val = cands_val
    return best_val

def largest_dmin_next(D, D_weights, k):
    # This algorithm mimics the 2-approximation of symmetric Max-Min Diversification.
    # Starts with an arbitrary point (we select it randomly), picks next point as the point with max d_min distance towards current set.
    l = list(D.nodes())
    v0 = np.random.choice(l, 1, replace=False)[0]
    sol = [v0]
    distances = {v: min(D[v][v0]['weight'],D[v0][v]['weight']) for v in D.nodes() if v not in sol}
    teller = 2
    while teller<=k:
        next_v = max(distances.items(), key = lambda x: x[1])[0]
        sol.append(next_v)
        distances = {v: min(distances[v], min(D[v][next_v]['weight'],D[next_v][v]['weight'])) for v in D.nodes() if v not in sol}
        teller +=1
    
    return div_score(D_weights, sol)

##########################################################################################
################ Help Functions ##########################################################
##########################################################################################

def cluster_dmax(D, D_weights, R):
    # The d_max clustering phase from our paper, with parameter R>0.
    # D_max is a NetworkX Graph, with edge attribute 'weights' the directed distances d_max (satisfying triangle ineq.).
    # Nodes are not reindexed from 0,1,2,... instead they use the same index as D.

    centers = set()
    marked = []
    check = True

    unmarked = set(D.nodes())
    while unmarked:
        c = unmarked.pop()
        centers.add(c)
        bad_v = set([v for v in unmarked if max(D[v][c]['weight'],D[c][v]['weight']) < R/3])
        unmarked.difference_update(bad_v)

    D_new = D.subgraph(centers)
    #D_weights_new = {k: D_weights[k] for k in D_new.edges()}
    return D_new

def create_auxiliary_graph(R, D_new, D_weights, k):
    # After the clustering phase, we create a graph if d_ij < R/6k.
    # k is the size parameter and R the guess of an optimum value.
    G_aux = nx.DiGraph()
    G_aux.add_nodes_from(D_new.nodes())
    #weight_dict = nx.get_edge_attributes(D_new,'weight')
    edges_filtered = [e for e in D_new.edges() if D_weights[e] < R/(6*k)]
    G_aux.add_edges_from(edges_filtered) # Note that these edges are unweighted!
    

    return G_aux

def div_score(D_weights, sol):
    # Computes the min-max objective function for a given set sol.
    # Distances defined by D, which a NetworkX Digraph, with edge attribute 'weights' the directed distances (satisfying triangle ineq.).
    # sol needs to be in D.nodes()
    node_pairs = list(combinations(sol, 2)) # this only generates (u,v) but not (v,u).
    score = min([min(D_weights[(u,v)],D_weights[(v,u)]) for (u,v) in node_pairs])

    return score

def optimum_MaxIndSet_binsearch(D, D_weights, k):
    # Guess the optimum R, and create a Max. Ind. Set problem.
    # unique_d = list(set([v for v in nx.get_edge_attributes(D,'weight').values()]))
    # This is way faster than brute-force.
    opt_val = 0
    unique_d = list(set([v for v in D_weights.values()]))
    unique_d.sort()
    a = 0
    b = len(unique_d)-1
    #weight_dict = nx.get_edge_attributes(D,'weight')
    while a<=b:
        midpoint = (a + b)//2
        R = unique_d[midpoint]
        #print(R)
        
        Gr = nx.Graph()
        Gr.add_nodes_from(D.nodes())
        #weight_dict = nx.get_edge_attributes(D,'weight')
        edges_filtered = [(u,v) for (u,v) in D_weights if D_weights[(u,v)] >= R and D_weights[(v,u)] >= R]
        Gr.add_edges_from(edges_filtered)
        clique = nx.max_weight_clique(Gr, weight=None)[0]

        if len(clique)>=k:
            cand = div_score(D_weights,clique)
            # try to look for a higher R value
            a = midpoint+1
            if cand > opt_val:
                opt_val = cand
                opt = clique
        else:
            b = midpoint-1      
    return opt_val

def optimum_MaxIndSet_binsearch2(D, D_weights, k):
    # Guess the optimum R, and create a Max. Ind. Set problem.
    # unique_d = list(set([v for v in nx.get_edge_attributes(D,'weight').values()]))
    # This is way faster than brute-force.
    opt_val = 0
    unique_d = list(set([v for v in D_weights.values()]))
    unique_d.sort()
    a = 0
    b = len(unique_d)-1
    #weight_dict = nx.get_edge_attributes(D,'weight')
    while a<=b:
        midpoint = (a + b)//2
        R = unique_d[midpoint]
        #print(R)
        
        Gr = nx.Graph()
        Gr.add_nodes_from(D.nodes())
        #weight_dict = nx.get_edge_attributes(D,'weight')
        edges_filtered = [(u,v) for (u,v) in D_weights if D_weights[(u,v)] >= R and D_weights[(v,u)] >= R]
        Gr.add_edges_from(edges_filtered)
        #clique = nx.max_weight_clique(Gr, weight=None)[0]
        found = False
        for clique in nx.find_cliques(Gr):
            if len(clique)>=k:
                found = True
                cand = div_score(D_weights,clique)
                # try to look for a higher R value
                a = midpoint+1
                if cand > opt_val:
                    opt_val = cand
                    opt = clique
                break
        if not found:
            b = midpoint-1      
    return opt_val
##########################################################################################
################ Finding Longest Shortest Path in a Directed Acyclic Graph (DAG) #########
##########################################################################################

# largest finite diameter:
# diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(Gc)])

def dag_shortest_path_source(Gc, topo_order, source):
    # Single source shortest paths to all other vertices, and returning the largest distance.
    # Runs in O(V+E) time, only works for DAGs.
    parent = {source: None}
    d = {source: 0}

    for u in topo_order:
        if u not in d: continue  # get to the source node
        for v in Gc.successors(u):
            if v not in d or d[v] > d[u] + 1:
                d[v] = d[u] + 1
                parent[v] = u

    # vertex with longest shortest path distance from source.
    max_v = max(d.items(), key = lambda x: x[1])[0]

    # Corresponding path.
    y = max_v
    path = []
    length_p = 0
    while y != None:
        path.append(y)
        y = parent[y]
        length_p += 1
    path.reverse()
    return path, max_v, length_p

def longest_shortest_path2(Gc, topo_order):
    # Iterating SS over all sources.
    l = deque(topo_order)
    n = len(topo_order)
    d_uv = {}
    for i in range(n-1):
        source = l[0]
        _, max_v, length_p = dag_shortest_path_source(Gc, l, source)
        d_uv[(source,max_v)] = length_p
        l.popleft()

    (u,v) = max(d_uv.items(), key = lambda x: x[1])[0] # the nodepair with largest (finite) distance.
    path, _, _ = dag_shortest_path_source(Gc, topo_order, u) # the corresponding path.

    return path


def dag_sp_secondway(Gc, topo_order):
    # Slower way.
    n = len(topo_order)
    parent_all = [{topo_order[i]: None} for i in range(n)] # list of dicts.
    d_all = [{topo_order[i]: 0} for i in range(n)] # list of dicts.

    tel1 = 1
    for u in topo_order:
        tel2 = 0
        for d in d_all[:tel1]:
            #print(u, d, tel2)
            tel2 += 1
            if u not in d: continue  # get to the source node
            for v in Gc.successors(u):
                if v not in d or d[v] > d[u] + 1:
                    d[v] = d[u] + 1
                    parent_all[tel2-1][v] = u    
        tel1 +=1

    max_keynval = [max(d.items(), key = lambda x: x[1]) for d in d_all]
    max_lists_vals = [x[1] for x in max_keynval]
    max_lists_keys = [x[0] for x in max_keynval]
    y = max(max_lists_vals)
    z = max_lists_vals.index(y)
    t = max_lists_keys[z]
    s = topo_order[z]

    # Corresponding path.
    path = []
    while t != None:
        path.append(t)
        t = parent_all[z][t]
    path.reverse()
    return path
    
        

##########################################################################################
################ Extracting a Chordless Cycle from DiGraph ##############################
##########################################################################################

# cycle_edgelist = nx.find_cycle(G, orientation="original")
# cycle = [e[0] for e in cycle_edgelist]

def shortcut_Rightmostneighbor(G, cycle, visited):
    detection = False
    H = G.subgraph(cycle).copy()
    u = cycle[0]
    visited.append(u)
    idx = {k: v for v, k in enumerate(cycle)}
    rightmost_ngb = list(H[u])[0]
    for v in H[u]:
            if idx[v]>idx[rightmost_ngb]:
                rightmost_ngb = v

    if rightmost_ngb in visited:
            detection = True
    next_cycle = cycle[idx[rightmost_ngb]:]
    next_cycle.append(u)
    return next_cycle, detection
    
def extract_chordless_cycle(G, cycle):
    # We extract a chordless cycle from a given cycle.
    detection = False
    next_cycle = cycle
    visited = []
    while not detection:
        next_cycle, detection = shortcut_Rightmostneighbor(G, next_cycle, visited)
                        
    return next_cycle

def chordless_cycle_check(G, c):
    b = False
    H = G.subgraph(c).copy()
    if len(H.edges())==len(c):
        b = True

    return b

def find_Chordless_cycle(G):
    # Find chordless cycle in directed graph G
    cycle_edgelist = nx.find_cycle(G)
    cycle = [e[0] for e in cycle_edgelist] 

    if not chordless_cycle_check(G, cycle):
        c = extract_chordless_cycle(G, cycle)
    else:
        c = cycle
    return c

##########################################################################################
################ Maximum Antichain in a DiGraph #####################
##########################################################################################

def lp_DAG_marked(G, topo_order, marked):
    # Outputs path with most *unmarked* vertices in a DAG g.
    # g is assumed to be a networkx graph, and marked a set of vertices in g.
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph is not a DAG!")

    if not set(marked).issubset(G.nodes()):
        raise ValueError("Marked vertices are not a subset of the graph's vertices!")
    
    dist = {} # largest number of unmarked vertices on a path ending at x.
    pred = {x: x for x in topo_order} # predecessors
    m = {} 
    for v in topo_order:
        if v in marked:
            m[v] = 0
            dist[v] = 0
        else:
            m[v] = 1
            dist[v] = 1
    
    for u in topo_order:
        for v in G.successors(u):
            if dist[v] < dist[u] + m[v]:
                dist[v] = dist[u] + m[v]
                pred[v] = u
    x = None
    y = max(dist, key=dist.get)
    lp = []
    while x != y:
        lp.append(y)
        x = y
        y = pred[y]
    lp.reverse()
    return lp

def naive_Initialization(g, topo_order):
    # The path cover simply consits of every singleton node.
    pc = [[v] for v in g.nodes()]
    sparse_g = g.copy()

    edge_counter = defaultdict(int) # counter the number of paths in the cover each edge is part of.
    node_counter = defaultdict(int) # counter the number of paths in the cover each node is part of.
    source_nodes = defaultdict(int) # counting how many paths start at a certain node
    sink_nodes = defaultdict(int) # counting how many paths end at a certain node
    for v in g.nodes():
        node_counter[v] = 1
        source_nodes[v] = 1
        sink_nodes[v] = 1

    return pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes
def sparsify_and_GreedyPaths(g, topo_order):
    # Algorithm 4 in 'MPC: The Power of Parameterization'-paper.
    # Simultaneous sparsifies (deleting transitive edges) and finds a greedy path cover.
    # G needs to be a DAG.
    marked = set()
    pc = [] # The cover will be a list of lists.
    n = len(topo_order)
    edge_counter = defaultdict(int) # counter the number of paths in the cover each edge is part of.
    node_counter = defaultdict(int) # counter the number of paths in the cover each node is part of.
    source_nodes = defaultdict(int) # counting how many paths start at a certain node
    sink_nodes = defaultdict(int) # counting how many paths end at a certain node
    sparse_g = g.copy()
    
    while len(marked) != n:
        lp = lp_DAG_marked(sparse_g, topo_order, marked)
        marked.update(lp)
        pc.append(lp)
        # All edges in the lp to the edge counter.
        for a, b in zip(lp, lp[1:]):
            edge_counter[(a,b)] += 1
            node_counter[a] += 1
        node_counter[lp[-1]] += 1
        source_nodes[lp[0]] += 1
        sink_nodes[lp[-1]] += 1
        
        R = set()
        for v in lp:
            for u in list(sparse_g.predecessors(v)):
                if u in R and (u,v) not in edge_counter:
                    sparse_g.remove_edge(u,v)
                    #print(u,v)
                    #print('Above edge deleted!')
                R.add(u) 
    return pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes

def create_MaxFlowInstance(n, topo_order, pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes):
    # We create the max. flow instance from the folklore reduction, described in 'MPC: The Power of Parameterization'-paper. 
    # G_orig is the initial graph is the original umodified graph (needs to be a DAG), edge_counter_dict and node_counter are used to give an initial path cover (and hence initial flow).
    # NodeIDs are assumed as follows: the v_in nodes are (0,1,...n-1), the v_out nodes are (n,n+1,...2n-1).
    # We assume original G_orig nodes are also (0,1,2,...). The SCC algorithm does this.
    # pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes = naive_Initialization(G_orig, topo_order)
    number_of_paths = len(pc)

    # Residual Graph
    G_r = nx.DiGraph()
    G_r.add_nodes_from(range(2*n))
    G_r.add_nodes_from(['s', 't'])

    # Every starting node of a path gets 1 flow from source.
    G_r.add_weighted_edges_from([('s', v, source_nodes[v]) for v in source_nodes])
    #G_r.add_weighted_edges_from([(v, 's', number_of_paths) for v in range(n)])

    # Every ending node of a path exists 1 flow to sink
    G_r.add_weighted_edges_from([(v+n, 't', sink_nodes[v]) for v in sink_nodes])
    #G_r.add_weighted_edges_from([('t',v, number_of_paths) for v in range(n,2*n)])

    # Edges between copies of the same nodes
    G_r.add_weighted_edges_from([(v, v+n, node_counter[v]-1) for v in node_counter if node_counter[v]>1])
    G_r.add_weighted_edges_from([(v+n, v, number_of_paths) for v in range(n)])

    # Edges between v_out and v_in nodes
    G_r.add_weighted_edges_from([(v+n,u, edge_counter[(v,u)]) for (v,u) in edge_counter])
    G_r.add_weighted_edges_from([(u,v+n, number_of_paths) for (v,u) in sparse_g.edges()])

    nx.set_edge_attributes(G_r, nx.get_edge_attributes(G_r, 'weight'), 'capacity')

    return G_r

def create_finalResidual(sparse_g, source_nodes, sink_nodes, edge_counter, node_counter, flow_dict, n):
    # Here we create the residual graph G_res corresponding to an optimal min_flow solution
    # To create the min_flow f_min = f_initial - f_maxflow, we update our flow with the max flow given by flow_dict.
    # Note that flow_dict is a dict of dicts (see nx.maximum_flow output).
    # To find the maximum antichain, we only care about reachability from 's' in G_res, so we dont put any capacities on the edges in G_res.

    # Initialize
    G_f = nx.DiGraph()
    G_f.add_nodes_from(range(2*n))
    G_f.add_nodes_from(['s', 't'])

    # Update flow leaving the source;
    for (u,f) in [(k,v) for k,v in flow_dict['s'].items() if v>0]:
        source_nodes[u] -= f
        node_counter[u] -= f

    # Update outgoing flow of v_in nodes. All are back-edges, so we add flow! only (v, v+n) is forward edge.
    for v in range(n):
        for (u,f) in [(k,y) for k,y in flow_dict[v].items() if y>0 and k != v+n]:
            edge_counter[(u % n,v)] += f
            node_counter[u % n] += f
            node_counter[v] += f
        if v+n in flow_dict[v]:
            # Forward edge
            node_counter[v] -= flow_dict[v][v+n]

    # Update outgoing flow of v_out nodes (also going to sink). All are forward edges, except (v,v-n) is backwards.
    for v in range(n,2*n):
        for (u,f) in [(k,y) for k,y in flow_dict[v].items() if y>0 and k != v-n]:
            node_counter[v % n] -= f
            if u != 't':
                edge_counter[(v % n,u)] -= f
                node_counter[u] -= f
            else:
                sink_nodes[v % n] -= f
        if v-n in flow_dict[v]:
            # Back edge
            node_counter[v % n] -= flow_dict[v][v-n]

    # Residual Graph
    G_f = nx.DiGraph()
    G_f.add_nodes_from(range(2*n))
    G_f.add_nodes_from(['s', 't'])

    # Every starting node of a path gets 1 flow from source.
    G_f.add_edges_from([('s', v) for v in source_nodes if source_nodes[v]>0])
    #G_r.add_weighted_edges_from([(v, 's', number_of_paths) for v in range(n)])

    # Every ending node of a path exists 1 flow to sink
    G_f.add_edges_from([(v+n, 't') for v in sink_nodes if sink_nodes[v]>0])
    #G_r.add_weighted_edges_from([('t',v, number_of_paths) for v in range(n,2*n)])

    # Edges between copies of the same nodes
    G_f.add_edges_from([(v, v+n) for v in node_counter if node_counter[v]>1])
    G_f.add_edges_from([(v+n, v) for v in range(n)])

    # Edges between v_out and v_in nodes
    G_f.add_edges_from([(v+n,u) for (v,u) in edge_counter if edge_counter[(v,u)]>0])
    G_f.add_edges_from([(u,v+n) for (v,u) in sparse_g.edges()])
    
    return G_f

def maxAntiChain(G_orig, topo_order):
    # We compute the maximum antichain, by looking at nodes reachable from 's' in the updated G_res.
    n = len(topo_order)
    pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes = naive_Initialization(G_orig, topo_order)
    # pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes = sparsify_and_GreedyPaths(G_orig, topo_order)
    G_r = create_MaxFlowInstance(n, topo_order, pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes)
    flow_value, flow_dict = nx.maximum_flow(G_r, 's', 't')
    G_f = create_finalResidual(sparse_g, source_nodes, sink_nodes, edge_counter, node_counter, flow_dict, n)
    cands = set(nx.descendants(G_f,'s'))
    cands_n = set.intersection(cands,range(n))
    maxAntichain = set()
    for x in cands_n:
        if x+n not in cands:
            maxAntichain.add(x)
    maxAntichain = list(maxAntichain)     
    return maxAntichain

##################### OLD CODE #####################

##def longest_shortest_path(Gc, topo_order):
##    # Here we compute the diameter (largest finite distance) of a DAG.
##    d_uv = {(u,v): 1 for (u,v) in Gc.edges()} # a dict d[(u,v)] shortest path distances.
##    link_uv = {(u,v): v for (u,v) in Gc.edges()}
##    reachable =  {u: set([v for v in Gc.successors(u)]) for u in topo_order}
##    for u in reversed(topo_order):
##        succ = list(Gc.successors(u))
##        reachable[u].update(*[reachable[v] for v in succ])
##        for w in reachable[u]:
##            check_list = [d_uv[(v,w)] for v in succ if (v,w) in d_uv]
##            if check_list:
##                min_val = min(check_list)
##                d_uv[(u,w)] = min_val + 1
##                idx = check_list.index(min_val) 
##                link_uv[(u,w)] = succ[idx]
##
##    (u,v) = max(d_uv.items(), key = lambda x: x[1])[0] # the nodepair with largest (finite) distance.
##    s = u
##    path = [s]
##    while s != v:
##        s = link_uv[(s,v)]
##        path.append(s)
##    
##    return d_uv, path

##def greedyPathCover(g, topo_order):
##    # Returns a path cover by greedily picking the path with most unmarked vertices.
##    # g needs to be a DAG.
##    marked = set()
##    pc = [] # The cover will be a list of lists.
##    n = len(topo_order)
##    edge_counter_dict = defaultdict(int) # counter the number of paths in the cover each edge is part of.
##    while len(marked) != n:
##        lp = lp_DAG_marked(g, topo_order, marked)
##        marked.update(lp)
##        pc.append(lp)
##
##        # All all edges in the lp to the edge counter.
##        for a, b in zip(lp, lp[1:]):
##            edge_counter_dict[(a,b)] += 1 
##
##    return pc, edge_counter_dict
##
##def updateResidual(G_r, n, flow_dict):
##    # Here we update the residual graph G_res with flow paths found in flow_dict.
##    # Note that flow_dict is a dict of dicts (see nx.maximum_flow output).
##    G_f = nx.DiGraph()
##    G_f.add_edges_from(G_r.edges(data = False))
##    # Update flow leaving the source;
##    for (u,f) in [(k,v) for k,v in flow_dict['s'].items() if v>0]:
##        G_f['s'][u]['weight'] = G_r['s'][u]['weight'] - f
##
##    # Update outgoing flow of v_in nodes. All are back-edges, so we add flow! only (v_in, v_out) is forward edge.
##    for v in range(n):
##        for (u,f) in [(k,y) for k,y in flow_dict[v].items() if y>0 and k != v+n]:
##            # Back-edges
##            if (u,v) in G_r.edges():
##                G_f[u][v]['weight'] =  G_r[u][v]['weight'] + f
##            else:
##                G_f.add_edge(u,v,weight=f)
##        if v+n in flow_dict[v]:
##            # Forward edge
##            G_f[v][v+n]['weight'] = G_r[v][v+n]['weight'] - flow_dict[v][v+n]
##
##    # Update outgoing flow of v_out nodes (also going to sink). All are forward edges, except (v_out, v_in) is backwards.
##    for v in range(n,2*n):
##        for (u,f) in [(k,y) for k,y in flow_dict[v].items() if y>0 and k != v-n]:
##            # Forward edges
##            if (v,u) in G_r.edges():
##                G_f[v][u]['weight'] = G_r[v][u]['weight'] - f
##        if v-n in flow_dict[v]:
##            # Possible back-edges
##            if (v-n,v) in G_r.edges():
##                G_f[v-n][v]['weight'] = G_r[v-n][v]['weight'] + flow_dict[v][v-n]
##            else:
##                G_f.add_edge(v-n,v,weight=flow_dict[v][v-n])
##    weighted_edges = nx.get_edge_attributes(G_f,'weight')
##    bad_edges = [e for e,w in weighted_edges.items() if w<=0]
##    G_f.remove_edges_from(bad_edges)
##    return G_f


##def optimum_bruteforce(D, k):
##    # We brute-force search for the optimum by checking all k-sized subsets of D.nodes()
##    opt_val = 0
##    for i in combinations(D.nodes(), k):
##        sol = list(i)
##        cand = div_score(D,sol)
##        if cand > opt_val:
##            opt_val = cand
##            opt = sol
##            
##    return opt_val
##
##def optimum_MaxIndSet(D, k):
##    # Guess the optimum R, and create a Max. Ind. Set problem.
##    # unique_d = list(set([v for v in nx.get_edge_attributes(D,'weight').values()]))
##    # This is way faster than brute-force.
##    opt_val = 0
##    unique_d = list(set([v for v in nx.get_edge_attributes(D,'weight').values()]))
##    for R in unique_d:
##        Gr = nx.Graph()
##        Gr.add_nodes_from(D.nodes())
##        weight_dict = nx.get_edge_attributes(D,'weight')
##        edges_filtered = [(u,v) for (u,v) in weight_dict if weight_dict[(u,v)] >= R and weight_dict[(v,u)] >= R]
##        Gr.add_edges_from(edges_filtered)
##        clique = nx.max_weight_clique(Gr, weight=None)[0]
##
##        if len(clique)>=k:
##            cand = div_score(D,clique)
##            if cand > opt_val:
##                opt_val = cand
##                opt = clique
##            
##    return opt_val


##

def printApprox():
    for k in range(2,len(D.nodes())+1):
        print("k = "+str(k))

        start = time.time()
        x = optimum_MaxIndSet_binsearch(D, D_weights, k)
        end = time.time()
        print("Optimal value = "+str(x)+" [Time (s) = "+str(round(end-start))+"]")

        start = time.time()
        x = approx1(D, D_weights, k)
        end = time.time()
        print("Approx1 solution = "+str(x)+" [Time (s) = "+str(round(end-start))+"]")

        start = time.time()
        x = approx2(D, D_weights, k)
        end = time.time()
        print("Approx2 solution = "+str(x)+" [Time (s) = "+str(round(end-start))+"]")

        start = time.time()
        x = approx3(D, D_weights, k)
        end = time.time()
        print("Approx3 solution = "+str(x)+" [Time (s) = "+str(round(end-start))+"]")

        start = time.time()
        x = randomk(D, D_weights, k, 10)
        end = time.time()
        print("RANDOM solution = "+str(x)+" [Time (s) = "+str(round(end-start))+"]")
        
        start = time.time()
        x = largest_dmin_next(D, D_weights, k)
        end = time.time()
        print("LargestNext solution = "+str(x)+" [Time (s) = "+str(round(end-start))+"]")
        print("**************")
    return 
