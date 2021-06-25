import collections

p = collections.OrderedDict()

p.n_sample = 64               # particles filters for evaluation
p.sz_T = [32, 32]
p.p_ini = 45
p.n_ini = 200
p.dict_sz = 40           # num of atoms in dictionary
p.p_up_num = 5           # num of pos samples updating per frame
p.n_up_num = 20          # num of neg samples updating per frame
p.recSp = 3
p.up_int = 10
p.feat_dim = 512
p.debug = False