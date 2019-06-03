from __future__ import print_function
from itertools import groupby
from operator import itemgetter
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
import sys,  writeFact, os, pandas as pd, numpy as np, itertools
import networkx as nx
import matplotlib.pyplot as plt


def head_to_children_dict(hid_cid):
    new_dict= {}
    for i in hid_cid:
        if i[0] in new_dict:
            new_dict[i[0]].append(i[1])
        else:
            new_dict[i[0]]= [i[1]]
    return(new_dict)

def reverse_tuple_list(cid_hid):
    hid_cid=[]
    for i,j in cid_hid:
        hid_cid.append((j,i))
#    hid_cid_new = hid_cid.copy()
#     for k in end_nodes:
#         hid_cid_new.append((k, -1))
    return(hid_cid)
    

def show_parse_information(relation_df):
    PID = relation_df['PID'].tolist()
    POS = relation_df['POS'].tolist()
    WORD = relation_df['WORD'].tolist()
    PIDWITH = relation_df['PIDWITH'].tolist()
    RELATION = relation_df['RELATION'].tolist()
    
    #filecontent = open(rawFile,"r").read()
    #filename = rawFile.split('/')[-1]

    #print (color.BOLD + folder_name+"/" + filename,": ", filecontent, end='' + color.END) 
    #print (color.BOLD + folder_name+"/" + filename,": ", filecontent, end='' + color.END)
    horiz = [('PID',PID),('WORDS',WORD)]
    horiz_df = pd.DataFrame.from_items(horiz)
#     horiz_df.style.apply(highlight_if_AUX, axis=0)
    horiz_df = horiz_df.set_index('PID').T
    print(horiz_df)
    print("==============================================")
    pidwith1 = tree.remove_duplicates_from_list(PIDWITH)
    print("Internal nodes & root node: ",pidwith1)
    end_nodes = tree.list_substraction(PID,pidwith1) 
    end_nodes_list = [[i] for i in end_nodes]
    print("End nodes(",len(end_nodes),") :",end_nodes)

#input: 2, output: 3.. where node 3 is head of node 2
def find_head_node(node, cid_hid):
    if node == 0:
        return ('0')
    else:
        head=[x[1] for i,x in enumerate(cid_hid,1) if i == node][0]
    return(head)

#input: 3, output:[1,2,4].. where node [1,2,4] are children of node 3
def find_children (node, hid_cid):
    children = []
    for i, x in enumerate(hid_cid, 1):
        if(node == x[0]):
            children.append(x[1])
    return (children)

#input: 1, output:[1,2,4].. where node [1,2,4] are siblings of node 1 (assumption, 1 is sibling of itself)    
def find_sibling (node, cid_hid, hid_cid):
    head = find_head_node(node, cid_hid)
    siblings = find_children (head, hid_cid)
    head = find_head_node(node, cid_hid)
    siblings_with_head = siblings + [head]
    if len(siblings)==0:
        siblings=[node]
    return (siblings, siblings_with_head)

#input 1, output: [1,3,6,7]...  where 7 is root of tree, (7-> 6), (6->3), (3->1)  and (a-> b) represents b is child of a
def go_till_root(node_id, cid_hid, rec):
    head = find_head_node(node_id, cid_hid)
    if head!=0:
        rec.append(head)
    if head == 0:
        return([0])
    else:
        head = go_till_root(head, cid_hid,rec)
    return(rec)

# def go_till_leaf(node_id, hid_cid, rec):
    child = find_children(node_id, hid_cid)
    if child!=0:
        rec.append(child)
    if child == 0:
        return([0])
    else:
        child = go_till_leaf(child, hid_cid ,rec)
#     print(rec)
    return(rec)

# list_tuple = [(1, 'aa'), (2,'bb'),..], node_id = 1, return value= ['aa']
# functionality as a dictionary for a list of tuple
def access_list_of_tuple(node_id , list_tuple):
    return([x[1] for i, x in enumerate(list_tuple,1) if x[0] == node_id])

# list_tuple = [(1, 'aa'), (2,'bb'),..], node_id = [1,2], return value= ['aa','bb']
# takes a list and applies dictionary on that elements of lists, only difference is, here dictionary is list of tuple
def access(node_id_list, list_tuple):
    node_val_list = [access_list_of_tuple(i, list_tuple)[0] for i in node_id_list]
#     print("access: ",node_val_list)
    return (node_val_list)

# makes union of 2 lists (merge 2 lists without duplicateions) eg [1,2,3,4] + [2,3,5] => [1,2,3,4,5]
def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return (final_list) 

# removes duplicate entries from list
def remove_duplicates_from_list(u):
    u.sort()
    u1 = list(u for u, _ in itertools.groupby(u))
    return (u1)

#subtract lst2 from lst1, note: noth lst1, lst2 should not have duplications
def list_substraction(PID,pidwith1):
    return(list(set(PID) - set(pidwith1)))

#parameters: i="VERB",j=POS; return value: list of ids of elements whose POS is VERB
def extract_id_of_i_value_from_j_column_list(i,j):
    result = [k for k, x in enumerate(j,1) if x == i ]
    return(result)

# extract respective pos/word mapping for all ids present 1st and only one column of df, where list_tuple contains wid-pos/word mapping for all wids
def extract_values_of_lists_from_df(df, list_tuple):
    result=[]
    for idx, row in df.iterrows():
        result.append(access(list(row)[0], list_tuple))
    return(result)
    
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

#input: [[2, 6, 7], [12, 13], [21, 22], [1]]
#output: ['2 6 7', '12 13', '21 22', '1']
def convert_int_elements_to_string(list_of_paths):
    a=[];changed=[]
    
    for every in list_of_paths:
        changed = [str(i) for i in every]
        joined = " ".join(changed)
        a.append(joined)
    return(a)
    
#Draws a tree for dataframe created from conll
def draw_tree(relation_df, cid_hid, which_lang, tmpSentPath):
    PID = relation_df['PID'].tolist()
    POS = relation_df['POS'].tolist()
    WORD = relation_df['WORD'].tolist()
    PIDWITH = relation_df['PIDWITH'].tolist()
    RELATION = relation_df['RELATION'].tolist()


    edges={}
    for i in zip(cid_hid, RELATION):
        edges[i[0]]=i[1]
    
    G = nx.DiGraph()

    G.add_nodes_from(PID)
    G.add_edges_from(cid_hid)

    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True, edge_color='black', node_color='yellow')
    nx.draw_networkx_edge_labels(G,pos,edge_labels= edges,font_color='red')


    #plt.savefig(tmpSentPath + '/' + folder_name +'_'+ which_lang +"_tree.png")
    plt.savefig( tmpSentPath + which_lang +"_tree.png")
    #plt.show()
    

def debug(expression):
    frame = sys._getframe(1)
#     print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))    
    return(expression)
# xxx = 3
# debug('xxx')


#u=[[2],[3],[6],[8],[10],[11],[13],[15]], wid_word_list=[(2,'bAxa'),(3,'jEse-jEse')...], [(2,'ADP'),(3,'DET')]
#output: df containing 3 columns: 1st: u in verticle, 2nd: corresponding words, 3rd: corresponding pos
#NOTE= u is list of list, the elements are lists hence we can sent u as [[1,2],[3,4]..] like this also.
#def show_in_df(wid_all, u, wid_word_list, wid_pos_list, wid_rel_list):
def show_in_df(u, wid_word_list, wid_pos_list, wid_rel_list):

    #     u = end_nodes_list #+ final_head + final_rec + final_siblings
#     u = remove_duplicates_from_list(u) #this should be done b4 callinf this function
    length = [len(i) for i in u]
        
    final_tuple = []
    for i in range(0, len(u)):
        final_tuple.append((length[i], u[i]))
    
    sorted_final_tuple = sorted(final_tuple, key=lambda x: x[0])
    
    final_list = [i[1] for i in sorted_final_tuple]
    
    for_df = [('#_of_paths', u)]
    df = pd.DataFrame.from_items(for_df)
    df.index = df.index + 1
    
       
    result_pos = extract_values_of_lists_from_df(df, wid_pos_list)
    result_word = extract_values_of_lists_from_df(df, wid_word_list)
    result_rel = extract_values_of_lists_from_df(df, wid_rel_list)
    
#     print(len(wid_all), wid_all)
#     print(len(result_pos), result_pos)

#     df['wid'] = wid_all
    df['pos'] = result_pos
    df['word'] = result_word
    df['relation'] = result_rel
    df.name = 'u'
    return(df)

# given a lst from range x to y, it returns all elements which are > x and < y but not in the lst
def find_missing(lst): 
    return [x for x in range(lst[0], lst[-1]+1) if x not in lst] 

#take a node_id and return all of its successors till leaf nodes in a sequencial order.
# def all_successor_of_node(node_id,  cid_hid, hid_cid):
#     siblings, siblings_with_head  = find_sibling(node_id, cid_hid, hid_cid)
# #     print(node_id)
#     siblings_with_head.sort()
# #     print(len(siblings_with_head),siblings_with_head ,siblings_with_head[0],siblings_with_head[-1])
#     missing = find_missing(siblings_with_head)
#     final_grouping = siblings_with_head + missing
#     final_grouping.sort()
#     print(node_id, final_grouping)
#     print("===")
    
# takes all recursive paths from top to bottoom and end_nodes and returns list of internal nodes and all group of internal node(i.e. internal node and all its successors))    
def successor_helper(correct_paths, end_nodes):
    
    all_groups =[]; internal_nodes=[]; corrected = []
    #length of every recursive path from node1 to node2, where node1 and node2 can be any node (leaf, internal, root)
    length = [len(i) for i in correct_paths]
    max_length = max(length)
#     print(correct_paths)
    
#     for item in correct_paths:
#         item.reverse()
#         if item[0] not in end_nodes:
#             internal_nodes.append(item[0])
#         internal_nodes=remove_duplicates_from_list(internal_nodes)
    
    for k in range(0, max_length+1):
#         print("for k:",k)
        child_of_0 =[]
        for item, l in zip(correct_paths, length):
            if item[-1] not in end_nodes:
                child_of_0.append(item[-1])
            child_of_0 = remove_duplicates_from_list(child_of_0)
#             print("new_internal: ", child_of_0)


            new = item[:-k]
            new.reverse()
            
            #corrected is subset of paths which removes paths like  [], [1], which will through error and not useful
            # simultaneously for grouping
            if(len(new)>1):
#                 print("=>",new)
                corrected.append(new)
                
            if len(new)>0:
            #from corrected the paths which doesn't start from leaf are discarded as those can't be valid group
#                 if new[0] not in end_nodes and new[0] not in internal_nodes:
                if new[0] not in end_nodes:
#                     print("internal_node_debug---:",new)
                    internal_nodes.append(new[0])
                internal_nodes = internal_nodes +  child_of_0
                internal_nodes=remove_duplicates_from_list(internal_nodes)
            
            
#         print("----------")
#     print(internal_nodes)
#     print(corrected)
    
    all_groups=[] ; 
    for inter in internal_nodes:
        tmp=[];
        for lst in corrected:
            if lst[0] == inter:
                tmp = tmp+ lst
#                 tmp.sort()
                tmp= remove_duplicates_from_list(tmp)
        all_groups.append(tmp)
    
    #temporary bug fixing: Bug(an empty list is coming in all_groups, remove that)
    all_groups = [x for x in all_groups if x != []]

    
    return (internal_nodes , all_groups)     
    
#take final_rec (all bottom to top recursive paths and list of leaf nodes and gives list of internal nodes and all group of internal node(i.e. internal node and all its successors))    
def successors(rec, end_nodes):
#     print(rec)
#     print(end_nodes)
    correct_paths = [] 
    
    for lst in rec:
        if lst[0] in end_nodes:
#             lst.reverse()
            correct_paths.append(lst)
#     print(correct_paths)
    internal_nodes, all_groups = successor_helper(correct_paths, end_nodes)
    return(internal_nodes, all_groups) 

#     call sibling function here and the find min and max number from it ... fill the void numbers in between,
#and those will be out final correct grouping in the tree.
     
#VAA+ sequences covered in this, called by leaf_node_to_internal_node_skewed_paths
def find_skew_path(path, internal_nodes, hid_cid):
#     print(path)
    internal_nodes_in_path = [x for x in path if x in internal_nodes]
    indices = [path.index(x) for x in path if x in internal_nodes]
#     print(internal_nodes_in_path)
    one_path=[]
    for i in range(0, len(path)):
#         print(path[i])
        children = find_children(path[i], hid_cid)
#             print("len:",len(children))
        
        if (len(children) < 2):
#             print(path[i])
            one_path.append(path[i])
#     print(one_path)
#     print("============")
    return(one_path)

#returns true if list contains consecutive sequence of numbers (either ascending or descending)
#not handling [10, 9, 11, 12] case yet from 2.9 (check whether it should be handled or not here)
def checkConsecutive(l): 
    return sorted(l) == list(range(min(l), max(l)+1))

#given a tuple of integer numbers returns list of all sequences =>i/p: (2,5) ==> o/p: [2,3,4,5] 
def expand_range_to_list(range_tuple):
    my_list = list(range(range_tuple[0], range_tuple[1] + 1))
    return(my_list)

#input is list in ascending order but with some missing values, returns list of subset in sequence with no missing values
# input: [3,6,10,11,12], output: [[10, 11, 12]]
def consecutive_subset(data):  
    ranges = []
    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
#         print(group)
        if len(group) >2:
            lst = expand_range_to_list((group[0],group[-1]))
#             print("lst:  :::::",lst)
            ranges.append(lst)
#     print(ranges)
    return(ranges)

#all skew subtrees and VAA+ sequences covered in this 
def leaf_node_to_internal_node_skewed_paths(internal_nodes,leaf_nodes, paths, hid_cid, wid_pos_list):
#     print("*****************************")
#     print("internal: ",internal_nodes)
#     print("leaf: ",leaf_nodes)
#     print("final_rec: ",paths)
#     print("original_rec length:",len(paths))
    valid1 =[]
    for every_path in paths:
#         print(every_path[0])
        if every_path[0] in leaf_nodes:
            valid1.append(every_path)
#     print("leaf to all internal length:",len(valid1))
#     print(valid1)
    child_to_root_path=[]; all_skewed = []
#     print("valid1: ",valid1)
#     print("=======================")
    
    #list other than skewed (internal node having more than one child), but in sequence
    for every in valid1:
        #not handling [10, 9, 11, 12] case yet from 2.9 (check whether it should be handled or not here)
#         print(every, checkConsecutive(every) )
        if (checkConsecutive(every)):
            all_skewed.append(every)
        if (checkConsecutive(every) == False and len(every)>3):
            new_every = every.copy()
            new_every.sort()
            new_every = consecutive_subset(new_every)
            if len(new_every) !=0:    #path containing no sequence at all
#                 print(every, new_every[0])
                all_skewed.append(new_every[0])
            
    
#     print("all skewed: ", all_skewed)
    for every_path in valid1:
#         print(every_path, len(every_path))
        child_to_root_path.append(every_path)
        one_skew = find_skew_path(every_path, internal_nodes, hid_cid)
        if len(one_skew) > 1:
            all_skewed.append(one_skew)
#     print("leaf to internal all skewed paths: ",all_skewed)
    all_skewed = remove_duplicates_from_list(all_skewed)
    return(all_skewed)

#find all paths containing V (called by tree())
def leaf_and_its_head(final_head, end_nodes, internal_nodes):
    VA=[]
    for item in final_head:
        if item[0] in end_nodes and item[1] in internal_nodes:
            VA.append(item)
#             print(item)
    return(VA)

#convert a list of tuples into dictioanry (called by find_all_lwg())
def Convert(tup, di): 
    for a, b in tup: 
        di.setdefault(a, []).append(b) 
    return di 

#find all correct local lwg from [skewed + leaf-head-pairs] for a given POS from parser.
#tmp1 : list of all possible paths[skewed + leaf-head-pairs], which_POS: the POS type of a wors, whose lwg need to be found
def find_all_lwg( tmp1 , wid_pos_list, which_POS, cid_hid):
    pos_dictionary = {} 
    pos_dictionary = Convert(wid_pos_list, pos_dictionary)
    final_verb_cases=[]
    tmp1_head = Given_a_list_find_head_of_all(tmp1, cid_hid)
    for item, item_head in zip(tmp1, tmp1_head):
        if pos_dictionary[item_head][0] == which_POS:
            final_verb_cases.append(item)
    final_verb_cases = remove_duplicates_from_list(final_verb_cases)
    return(final_verb_cases)

# Takes a list of list and returns a list of head of every innner list
def Given_a_list_find_head_of_all(all_groups, cid_hid):
    all_heads=[];final_head=0
#     print(all_groups)
    for group in all_groups:
#         print(group)
        for i in group:
            head_of_i = find_head_node(i,cid_hid)
            if head_of_i in group:
                continue
            else:
                final_head = i
#         print(final_head)
        all_heads.append(final_head)
#     print(len(all_groups), len(all_heads))

    return(all_heads)

def find_single_AUX(pid, pos):
    l=[i for i,v in enumerate(pos) if v == 'AUX']
    
    
    all_single_aux = []
    for i,j in enumerate(l):
        x=[]
        if i==0:
            if pos[j+1]!='AUX':
                print(i, pos[j], pid[j])
#                 x = pid[j]
                x.append(pid[j])
        
        if i == len(l)-1:
            if pos[j-1]!='AUX' and pos[j-1]!='VERB':
                print(i, pos[j], pid[j])
                x.append(pid[j])
        if i > 0 and i < len(l)-1:
            if pos[j-1]!='AUX' and pos[j-1]!='VERB' and pos[j+1]!='AUX'  :
                print(i, pos[j], pid[j])
#                 x = pid[j]
                x.append(pid[j])
        single_aux = x.copy()  
        
        all_single_aux.append(single_aux)
        all_single_aux = remove_duplicates_from_list(all_single_aux)
    print("# of single auxes",all_single_aux)
    return(all_single_aux)   
    
