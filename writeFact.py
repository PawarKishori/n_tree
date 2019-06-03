# -*- coding: utf-8 -*-
import re
import operator
import sys,  writeFact, os, pandas as pd, numpy as np
import inspect

hindi_punct=['_','_',"'","!",'"',"#","$","%","&","'","(",")",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","]","^","_","`","{","|","}","~","'"]

#Takes a tree in dataframe format, maps all parser ids(pid) to word ids(wid) returns the dataframe of a tree containing wids.
def convertPIDsToWIDs(relation_df):
    relation_df.PID = relation_df.PID.replace(p_w)
    relation_df.PIDWITH = relation_df.PIDWITH.replace(p_w)
#     print(relation_df)
    
    relation_df = relation_df[~relation_df["PID"].astype(str).str.startswith('P', na=False)]
    relation_df = relation_df[~((relation_df["RELATION"].astype(str).str.startswith('punct', na=False)) & (relation_df["POS"].astype(str).str.startswith('PUNCT', na=False)))]
#     relation_df = relation_df[~relation_df["RELATION"].astype(str).str.startswith('punct', na=False)]
#     print(relation_df)
    return(relation_df)



#takes a tree in dataframe and returns list of tuple of edges in the tree
def extractUnlabelledDependency(relation_df):
    cid = relation_df['PID'].tolist() ;     hid = relation_df['PIDWITH'].tolist() ;     cid_hid=[]
    for i in range(0,len(cid)):
	#child_id-head_id
        cid_hid.append((cid[i], hid[i]))
    return(cid_hid)


def createH_wid_word_and_PunctFact(sent):
    with open(sent,"r") as f:
        i=1;wid_word_list=[];punctlist=[]; pattern=re.compile("\w+")
        wid_word_list.append((0,"wroot"))
        x=f.read()
        #print(x) #original sentence
        x=" ".join(x.split()) #removing more than one white spaces in line
        for word in x.strip().split(" "):
            #print(i,word)
            #if pattern.match(word):   # if word contains only punctuation eg.'=' exclude it
            #print(word[0],word)
            #print('('in string.punctuation)
            #print(word)
            #if (word[0] in string.punctuation or word[0] in hindi_punct) and len(word)>1: #left punct  
            if word[0] in hindi_punct and len(word)>1: #left punct  
                #print("left punct")
                punct=word[0]
                punctlist.append((punct,"L",i))
                word=word.lstrip(punct)
            #if word[-1] in string.punctuation and len(word)>1:
            if word[-1] in hindi_punct and len(word)>1:
                punct=word[-1]
                punctlist.append((punct,"R",i))
                if len(word)>=2:
                    #if word[-2] in string.punctuation:
                    if word[-2] in hindi_punct:
                        punct1=word[-2]
                        punctlist.append((punct1,"R",i))
                        word=word.rstrip(word[-1])
                word=word.rstrip(word[-1])
                #print(word) 
            wid_word_list.append((i,word))
            i+=1
        wid_word_dict={}
        for pair  in wid_word_list:
            #print(pair[0], pair[1])
            wid_word_dict[0]='root'
            wid_word_dict[pair[0]]=pair[1]
        
        return([wid_word_list,punctlist, wid_word_dict])

#from raudra parser1.py====================


def create_hindi_dataframe(parse):
    df= pd.read_csv(parse, sep='\t',names=['PID','WORD','1-','POS','2-','3-','PIDWITH','RELATION','4-','5-'])
    df.index = np.arange(1,len(df)+1)
    df1= df[['PID','WORD','POS','RELATION','PIDWITH']]
    pid = df1.PID.apply(lambda x : 'P'+str(x))
    pidwith = df1.PIDWITH.apply(lambda x : 'P'+str(x))
    relation_df =  pd.concat([pid, df1.WORD,df1.POS, df1.RELATION, pidwith], axis=1)
    return relation_df

def create_english_dataframe(parse):
    df= pd.read_csv(parse, sep='\t',names=['PID','WORD','WORD_ROOT','POS','POS_STANFORD','1-','PIDWITH','RELATION','2-','3-'])
    df.index = np.arange(1,len(df)+1)
    df1= df[['PID','WORD','WORD_ROOT','POS','POS_STANFORD','RELATION','PIDWITH']]
    pid = df1.PID.apply(lambda x : 'P'+str(x))
    pidwith = df1.PIDWITH.apply(lambda x : 'P'+str(x))
    relation_df = pd.concat([pid, df1.WORD, df1.WORD_ROOT, df1.POS, df1.POS_STANFORD, df1.RELATION, pidwith], axis=1)
    return relation_df

def create_english_facts(parse, wid_word_list):
    relation_df = create_english_dataframe(parse)
    
    [wid_pid,p_w, wid_pos_list]=writeFact.createWID_PID(wid_word_list,relation_df['PID'].tolist(),relation_df['WORD'].tolist(),relation_df['POS'].tolist())
    
    writeFact.add(wid_pid,"E_wid-pid",tmpSentPath+"/E_word_id_parser_id_mapping.dat")
    #writeFact.debug_check(tmpSentPath+"/E_word_id_parser_id_mapping.dat")
    writeFact.addLists([relation_df['PID'].tolist(),relation_df['WORD'].tolist()],"E_pid-word",tmpSentPath+"/E_parser_id_word_mapping.dat") 
    #writeFact.debug_check(tmpSentPath+"/E_parser_id_word_mapping.dat")
    writeFact.addLists([relation_df['POS'].tolist(),relation_df['RELATION'].tolist(),relation_df['PID'].tolist(),relation_df['WORD'].tolist(),relation_df['PIDWITH'].tolist()],"E_pos1-relation-pid1-word1-pid2",tmpSentPath+"/E_conll_facts.dat") 
    #writeFact.debug_check(tmpSentPath+"/E_conll_facts.dat")

    relation_df.PID = relation_df.PID.replace(p_w)
    relation_df.PIDWITH = relation_df.PIDWITH.replace(p_w)
    
    relation_df = relation_df[~relation_df["PID"].str.startswith('P', na=False)] 
    
    modified_pidwith=relation_df['PIDWITH'].tolist()
    modified_pid=relation_df['PID'].tolist()
    word_pidwith= [wid_word_dict[k] for k in modified_pidwith] #was working in python2
    word_pid= [wid_word_dict[k] for k in modified_pid]         #was working in python2
    
    writeFact.addLists([relation_df['PID'].tolist(),relation_df['WORD'].tolist(),relation_df['POS'].tolist(),relation_df['POS_STANFORD'].tolist(),relation_df['RELATION'].tolist(),relation_df['PIDWITH'].tolist(),word_pidwith],"E_pos1-pos_std1-relation-cwid-cword-hwid-hword",tmpSentPath+"/E_parse.dat")
#     writeFact.debug_check(tmpSentPath+"/E_parse.dat")
    cid_hid = extractUnlabelledDependency(relation_df)
    #print(cid_hid)
    
    
def create_hindi_facts(parse, wid_word_list,tmpSentPath ):
    relation_df = create_hindi_dataframe(parse)

    [wid_pid,p_w, wid_pos_list]=writeFact.createWID_PID(wid_word_list,\
            relation_df['PID'].tolist(),relation_df['WORD'].tolist(),relation_df['POS'].tolist())

    writeFact.add(wid_pid,"H_wid-pid",tmpSentPath+"/H_word_id_parser_id_mapping.dat")
    #writeFact.debug_check(tmpSentPath+"/H_word_id_parser_id_mapping.dat")
    writeFact.addLists([relation_df['PID'].tolist(),relation_df['WORD'].tolist()],"H_pid-word",tmpSentPath+"/H_parser_id_word_mapping.dat")
    #writeFact.debug_check(tmpSentPath+"/H_parser_id_word_mapping.dat")
    writeFact.addLists([relation_df['POS'].tolist(),relation_df['RELATION'].tolist(),relation_df['PID'].tolist(),relation_df['WORD'].tolist(),relation_df['PIDWITH'].tolist()],"H_pos1-relation-pid1-word1-pid2",tmpSentPath+"/H_conll_facts.dat")
    #writeFact.debug_check(tmpSentPath+"/H_conll_facts.dat")

    
    relation_df.PID = relation_df.PID.replace(p_w)
    relation_df.PIDWITH = relation_df.PIDWITH.replace(p_w)
#     print(relation_df)
    
    relation_df = relation_df[~relation_df["PID"].str.startswith('P', na=False)]
       
    writeFact.addLists([relation_df['POS'].tolist(),relation_df['RELATION'].tolist(),relation_df['PID'].tolist(),relation_df['WORD'].tolist(),relation_df['PIDWITH'].tolist()],"H_pos1-relation-cid-word1-hid",tmpSentPath+"/H_parse.dat")
    tam_lwg = writeFact.extract_tam_lwg_ids()

    cid_hid = extractUnlabelledDependency(relation_df)
    #print(cid_hid)
    #tree(relation_df, wid_word_list, cid_hid, wid_pos_list)
    #dff = for_anand(rawFile, relation_df)

    return(relation_df)


def checkLwgParseAgainstDefiniteLWG(relation_df,vib_lwg,tam_lwg, wid_word_list, cid_hid):
    print("hello")

#===========================parser1.py



def extract_tam_lwg_ids():
    filepath= '.'
    exists = os.path.isfile(filepath +'/'+'revised_manual_local_word_group.dat')
    if exists:
        with open(filepath+'/'+"revised_manual_local_word_group.dat", "r") as f_in:
            lines = list(line for line in (l.strip() for l in f_in) if line)   
    else:
        with open(filepath +'/'+"manual_local_word_group.dat","r") as f:
            lines1 = list(line for line in (l.strip() for l in f_in) if line)   
#     print(len(lines))
    if len(lines) == 0:
        data_list = lines1
        column_no = 3
    else:
        data_list = lines
        column_no = 5
           
    tam_lwg=[]      
    word_entry_list = [x.split("\t") for x in data_list]

#     print(word_entry_list, len(word_entry_list))
#     print("-------------")
    
    for i in (word_entry_list):
#         print(i, len(i), i[5], type(i[column_no]))
        if (i[5] == '0)'):
            continue
#             print(" zero")
        else:
#             print(i[5],"not zero")
            tam_lwg.append((i[5].rstrip(")")))
#     print(tam_lwg)
    return(tam_lwg)


def debug_check(filepath):
    #filepath='./revised_manual_local_word_group.dat'
    #filepath='./unttled.txt'
    filename= filepath.split("/")[-1]
    functionname = inspect.currentframe().f_code.co_name
    if (os.path.exists(filepath)):
        print("Created: "+ filename + "\n")
        if(os.path.getsize(filepath)==0):
            print(filename + " is empty, check => " + functionname+ "\n")
    else:
        print(filename + " is not created, check => " + functionname+ "\n")


def split_txt_to_dat(filepath, sub_dirname ):
    #filepath = '/home/raudra/codes/demo_codes_for_alignment/E_typed_dependency_parse.txt'
    filename = filepath.split('/')[-1].split('.')[:-1][0] +'.dat'
#     dir_path = filepath.split('/')[:-1][0]
    dir_path = "/".join(filepath.split('/')[:-1])
    #sub_dirname = '/home/raudra/codes/demo_codes_for_alignment/dir_names.txt'
    

    with open(filepath,"r") as f:
        text = f.read().split(";~~~~~~~~~~")
        text = text[:-1] #removing last empty line directory
        sen_dir_name = open(sub_dirname, "r").read().splitlines()
        print(sen_dir_name)
        #print(text)
        print(len(text), len(sen_dir_name))
#     print(text[5])
        if len(sen_dir_name) == len(text):
            for i in range(0, len(text)):
                sub_dir = dir_path+ "/" + sen_dir_name[i]
                dat_path = sub_dir + '/' + filename
                if (os.path.exists(sub_dir)):
                    with open( dat_path , "w") as g:
                        g.write(text[i])
                else:
                    print("Error trace: "+ sub_dir + " does not exists")
            
        else:
            print( "Error trace: "+ filepath + " and " + sub_dirname + "have varible number of sentences information..")



def add(fact_items,string,filename):
    #print("inside writefact.add")
    #print(fact_items)
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename,"a") as f:
        for line in fact_items:
            tokens=len(fact_items)
            fact="("+string
            for i,a in enumerate(line):
                fact=fact+"\t"+str(a)
                #print(fact)
            fact=fact+")\n"
            #print(fact)
            f.write(fact)

def addLists(factlist,factname,filename):
    #print("Inside writeFactLists Function")
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename,"a") as f:
        for i in range(0,len(factlist[0])):  #i= number of rows
            fact="("+factname
            for j in range(0,len(factlist)): #j=number of column
                fact=fact+"\t"+str(factlist[j][i])
            fact=fact+")\n"
            #print(fact)
            f.write(fact)
    #print("END: Inside writeFactLists Function")



p_w={}
def createWID_PID(wid_word, PID, PWORD, POS, RELATION):
    #print("******************** START createWID_PID ******************")
    #for x in range(0,len(PID)):
    #    print(PID[x],PWORD[x],POS[x]) 
    wid_pid = []; pid_wid=[];wid_pos_list=[]; wid_rel_list=[]
    index=0
    wid_pos_list.append((0,"ROOT"))
    wid_rel_list.append((0,"r_ROOT"))

    wid_pid.append((0,'P0'))
    pid_wid.append(('P0',0))
    p_w['P0']=0;new_punct_check=[]
    
    for item in wid_word:
        #print(item)
        for i in range(index,len(PWORD)):
            if item[1]==PWORD[i]:
                p_w[PID[i]]=item[0]
                wid_pid.append((item[0],PID[i]))
                pid_wid.append((PID[i],item[0]))
                wid_pos_list.append((item[0],POS[i]))
                wid_rel_list.append((item[0],RELATION[i]))
                index=i
                new_punct_check.append(item[0])
                break
    #print( len(wid_word),len(new_punct_check))
    #if len(wid_word)!=len(new_punct_check):
    #    print("WARNING: NEW PUNCTUATION OCCURED IN SENTENCE: CHECK H_wid-word.dat AND ADD THE NEW PUNCTUATION FROM WORD TO hindi_punct list(line 5) ")
    #print("******************** END createWID_PID ******************")
    return([wid_pid, p_w, wid_pos_list, wid_rel_list])



#===================================
def sortTuple(vib_list,order):
    if (order== 'D'):
        vib_list.sort(key = operator.itemgetter(1), reverse = True)
    if (order == 'A'):
        vib_list.sort(key = operator.itemgetter(1), reverse = False)
    return(vib_list)

def reFunction(vibPattern,sent):
    matches=len(re.findall(vibPattern,sent))
    indices=[(m.start(0), m.end(0)) for m in re.finditer(vibPattern, sent)]
    return([matches,indices])

def findWordPositionInSentence(tmpString, vibPattern):
    vibIds=[]
    vibStartId=len(re.findall(" ",tmpString))+1
    numOfWordsInVibhakti=len(re.findall(" ",vibPattern)) +1
    if numOfWordsInVibhakti == 1:
        vibIds.append(vibStartId)

    if numOfWordsInVibhakti >1 :
        #print("true")
        x=vibStartId
        for i in range(0,numOfWordsInVibhakti):
            x=x+i
            vibIds.append(x)
    return([vibStartId,numOfWordsInVibhakti,vibIds])

def lwg_of_postprocessors(wid_word,vibhaktis):
    words=[];wid=[]; sent_list=[]
    for i in wid_word:
        words.append(i[1])
        wid.append(i[0])
        #Creation of sentence without punctuation
        sent=" ".join(words)
        sent_list.append((i[1],i[0]))
    #print(sent)
    #print(sent_list)
    vib_list=[]

    for v in vibhaktis:
        n=len(v.split(" "))
        vib_list.append((v,n))
        #Sorting of vibh_list in descending order of word length
        vib_list=sortTuple(vib_list,'D')
    
    visited=[];item2WriteInFacts=[];item=[]
    all_vib_ids=[]
    
    for vib in vib_list:
        #print("======================================================================================================")
        vibPattern=vib[0]
        vibStartId=0;
        matches,indices=reFunction(vibPattern,sent)
        if (len(indices)!=0 ): #if the vib is not in sentence  but it is in vib list
            for IND in indices:                #if the same vibhakti is twice in sentence, for loop will be iterated twice
                new_vib_word=sent[IND[0]-1:IND[1]+1]
                if (" "+vibPattern+" " == new_vib_word):
                    tmpString=sent[:IND[0]]        # The whole sentence from start to vibhakti occurance starting index.
                    vibStartId, numOfWordsInVibhakti,vibIds = findWordPositionInSentence(tmpString,vibPattern)
                    #print("+++++++++++",tmpString,IND[0],vibStartId,vibIds)
                    new_vib_word_join="_".join(new_vib_word.strip(" ").split(" "))
                    vibStartId_str=str(vibStartId)
                    #print("B4 IF: ", vibStartId_str,visited)
                    if (vibStartId_str not in visited):
                        visited.append(vibStartId_str)
                        all_vib_ids.append(vibIds)
                        #print("After appending (vibStart, visited): ",vibStartId_str,visited)
                        noun_id=vibStartId-1  #8
                        noun=words[noun_id-1]   #words[8-1] = karane (since word_id = array indix +1)
                        lwg_ids=(noun_id,vibIds)
                        item.append(lwg_ids)
                        #print("noun_id, noun: ",noun_id,noun)
                        #print("item: ",item)
                        new_vib_word_strip="_".join(new_vib_word.strip(" ").split(" "))
                        #print("new_vib_word: ", new_vib_word_strip)
                        lwg="_".join([noun,new_vib_word_strip])
                        #print("lwg: ",lwg)
                        vibIds_str = [str(i) for i in vibIds] 
                        item2WriteInFacts.append((lwg,noun_id,noun," ".join(vibIds_str)))

    #print("All_vibhakti_IDs====")
    #print(all_vib_ids)
    #print(item2WriteInFacts)
    return([item2WriteInFacts,item , all_vib_ids])


#==================================================================================================

def flatten_list(list_of_list):
    final_list = [item for sublist in list_of_list for item in sublist]
    return final_list

