import numpy as np
import copy
from scipy.sparse import kron, identity

np.random.seed(seed=123) # for reproducibility

sym = lambda M: M + M.T
local_op = lambda d: sym(np.random.randn(d,d))
class Site():
    def __init__(self, c, rand=False):
        self.length = 1 # length
        self.ops = ops = {}
        J = c['J'] if 'J' in c.keys() else np.random.randn()
        Jz = c['Jz'] if 'Jz' in c.keys() else np.random.randn()
        alpha1 = c['alpha1'] if 'alpha1' in c.keys() else np.random.randn()
        alpha2 = c['alpha2'] if 'alpha2' in c.keys() else np.random.randn()
        beta = c['beta'] if 'beta' in c.keys() else np.random.randn()
        gamma = c['gamma'] if 'gamma' in c.keys() else np.random.randn()
        
        self.c = c = {'J':J, 'Jz':Jz, 'alpha1':alpha1, 'alpha2':alpha2, \
                          'beta':beta, 'gamma':gamma}
        self.all_couplings = list(c.values())
        
        # build operator dictionary
        ops["H"] = np.zeros((2,2)) # local fields are 0
        ops["Sxz"] = np.array([[0.5, 0], [0, -0.5]]) # z spin (S^z) operator
        ops["Sp"] = np.array([[0.0, 1.0], [0.0, 0.0]]) # raising (S^+) operator
        
        if rand:
            ops["Sxz"] = np.array([[c['alpha1'], 0], [0, -c['alpha2']]]) + \
                    np.array([[0, c['beta']], [c['beta'], 0]])
            ops["Sp"] *= c['gamma']
    
    def get_dim(self):
        return list(self.ops.values())[0].shape[0] # all ops should have same dimensionality
        
    def enlarge(self, site):
        '''Enlarge block by a single site'''
        
        D1, H1, Sxz1, Sp1 = self.get_dim(), self.ops['H'], self.ops['Sxz'], self.ops['Sp'] # this block
        D2, H2, Sxz2, Sp2 = site.get_dim(), site.ops['H'], site.ops['Sxz'], site.ops['Sp'] # another block (ie free site)

        enlarged = copy.deepcopy(self)
        enlarged.length += site.length
        enlarged.all_couplings += site.all_couplings
        ops = enlarged.ops

        ops['H'] = kron(H1, identity(D2)) + kron(identity(D1), H2) + self.interaction_H(site)
        ops['Sxz'] = kron(identity(D1), Sxz2)
        ops['Sp'] = kron(identity(D1), Sp2)

        return enlarged
    
    def interaction_H(self, site):
        '''Given another block, returns two-site term in the 
        Hamiltonain that joins the two sites.'''
        Sxz1, Sp1 = self.ops["Sxz"], self.ops["Sp"] # this block
        Sxz2, Sp2 = site.ops["Sxz"], site.ops["Sp"] # other block
        
        join_Sp = (self.c['J']/2)*(kron(Sp1, Sp2.conjugate().transpose()) + kron(Sp1.conjugate().transpose(), Sp2))
        join_Sxz = self.c['Jz']*kron(Sxz1, Sxz2)
        return (join_Sp + join_Sxz)
    
    def rotate_ops(self, transformation_matrix):
        # rotate and truncate each operator.
        new_ops = {}
        for name, op in self.ops.items():
            new_ops[name] = self.rotate_and_truncate(op, transformation_matrix)
        self.ops = new_ops
    
    @staticmethod
    def rotate_and_truncate(S, O):
        '''Transforms the operator to a new (possibly truncated) basis'''
        return O.conjugate().transpose().dot(S.dot(O)) # eqn 7 in arXiv:cond-mat/0603842v2

# set of utility functions for building and batching Hamiltonains

'''obtain an N-site Hamiltonian built from coupling coefficients c. If coupling coefficients
are not provided in dictionary c, they are drawn randomly from a normal distribution (mu=0, sigma=1)'''
def ham(N, c, rand=False):
    sys = Site(c, rand=rand)
    for _ in range(N-1):
        fs = Site(c, rand=rand)
        sys = sys.enlarge(fs)
    return sys.all_couplings, sys.ops['H'].todense()

'''obtain a batch of training data; return tuple includes the Hamiltonian, ground state,
ground state energy, coupling coefficients,  and optionally some low-lying excited states
and their energies'''
def next_batch(D, batch_size, just_ground=True):
    couplings = {'alpha1':1, 'alpha2':1, 'beta':1, 'gamma':1}
    c_list = [] ; H_list = [] ; e0_list = [] ; psi0_list = []
    e1_list = [] ; psi1_list = [] ; e2_list = [] ; psi2_list = []
    for _ in range(batch_size):
        c, H = ham(couplings, rand=True)
        e0, psi0 = eigsh(H,k=3, which="SA")
        c_list.append(c) ; H_list.append(np.asarray(H).ravel()) ; e0_list.append(e0[0].ravel()) ; psi0_list.append(psi0[:,0].ravel())
        if not just_ground:
            e1_list.append(e0[1].ravel()) ; psi1_list.append(psi0[:,1].ravel())
            e2_list.append(e0[2].ravel()) ; psi2_list.append(psi0[:,2].ravel())
    out = (np.vstack(c_list), np.vstack(H_list), np.vstack(e0_list), np.vstack(psi0_list))
    if not just_ground:
        extras = (np.vstack(e1_list), np.vstack(psi1_list), np.vstack(e2_list), np.vstack(psi2_list))
    out = out if just_ground else out + extras
    return out

'''obtain a random normalized state of dimension d'''
def rand_psi(d):
    psi = np.random.rand(1, d)
    psi /= np.sqrt(np.dot(psi, psi.T))
    return psi