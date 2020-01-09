import numpy as np
import time
import ipdb
import torch
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt

import itertools
def deep_iter(*shape):
    iters = (range(i) for i in shape)
    return itertools.product(*iters)
        
        

class Pde:
    def __init__(
            self,
            dim=1,
            lam=0.0,
            drift = lambda s,a: [0.]*self.dim,
            run_cost = lambda s,a: float(-self.dim),
            term_cost = lambda s: sum(map(lambda a: 0.5*a**2 - 0.5*a, s)),
            limit_s = 1.0, #l-infinity limit for state
            limit_a = 0, #l-infinity limit for action
            verbose=True
    ):
        self.dim = dim
        self.lam = lam
        self.drift = drift
        self.run_cost = run_cost
        self.term_cost = term_cost            
        self.limit_s = limit_s
        self.limit_a = limit_a

        if verbose:
            print(str(dim) + '-dim HJB')
    
    #domain is a unit hyper cube        
    def is_interior(self, s):
        return all(0<s<1)
    
    #cfd2mdp
    def mdp(self, n_mesh_s = 8, n_mesh_a = 16, method='cfd'):
        out = {}
        
        ####domain of mdp
        h_s = self.limit_s/n_mesh_s #mesh size in state
        h_a = self.limit_a/n_mesh_a #mesh size in action
        self.v_shape = tuple([n_mesh_s + 1]*self.dim)
        self.a_shape = tuple([n_mesh_a + 1]*self.dim)
        
        def is_interior(*ix_s):
            return all([0<x<n_mesh_s for x in ix_s])
        
        out.update({
                'v_shape': self.v_shape,
                'a_shape': self.a_shape,
                'is_interior': is_interior
                })
        ####domain
 
       # convert index(tuple) to state
        def i2s(*ix): 
            return np.array([x * h_s for x in ix])       
        out['i2s'] = i2s
        #convert index to action
        def i2a(*ix):
            return np.array([x * h_a for x in ix])
        #out['i2a'] = i2a


       
        ########running and terminal costs and discount rate
        def run_cost(ix_s,ix_a):
            return self.run_cost(i2s(*ix_s), i2a(*ix_a))*h_s**2/self.dim
        
        def term_cost(ix_s):
            return self.term_cost(i2s(*ix_s))
        
        self.rate = self.dim/(self.dim+self.lam*(h_s**2))
        out.update({
                'run_cost': run_cost,
                'term_cost': term_cost,
                'rate': self.rate
                })
        #########
        
        #####transition
        #return:
        #   a list of nbd indices
        #   a list of prob
        def step(ix_s, ix_a):
            ix_next_s_up = (np.array(ix_s)+np.eye(self.dim)).astype(int).tolist()
            ix_next_s_dn = (np.array(ix_s)-np.eye(self.dim)).astype(int).tolist()
            ix_next_s = [tuple(ix) for ix in ix_next_s_up+ix_next_s_dn]
            
            pr=[]
            if method == 'cfd':
                b = self.drift(i2s(*ix_s), i2a(*ix_a))
                pr_up = ((1+2.*h_s*b)/self.dim/2.0).tolist()
                pr_dn = ((1-2.*h_s*b)/self.dim/2.0).tolist()
                pr = pr_up+pr_dn
            
            return ix_next_s, pr, run_cost(ix_s,ix_a)
        out.update({'step': step,
                   'dim': self.dim})

        
        def bellman(self, ix, ia, v):
            s = self.i2s(ix)
            disc = self.rate
            ix_next, pr_next,run_h = self.step(ix,ia)
            lhs = v(torch.FloatTensor(s)); rhs = 0.
            #ipdb.set_trace()
            if self.is_interior(ix):            
                rhs += run_h 
                for ix1, pr1 in zip(ix_next, pr_next):
                    rhs += pr1*v(torch.FloatTensor(self.i2s(ix1)))
                rhs *= disc
            else:
                rhs = self.term_h(ix)
            return (rhs - lhs)
        
           out.update({'bellman': bellman})
    
        return out

def solver(mdp, n_epoch = 500):
    ######### nn for value
    # Linear regression model
    value = nn.Sequential(
        nn.Linear(mdp['dim'], 2*mdp['dim']+10),
        nn.ReLU(),
        nn.Linear(2*mdp['dim']+10, 1),
    )   
    print(value)
    
    #loss
    def tot_loss():
        out = 0.
        for ix in deep_iter(*mdp['v_shape']):
            ia = tuple([0]*mdp['dim'])
            out += mdp['bellman'](ix,ia,value)**2
        return out#/mdp.v_size_
    
    print_n = 10
    epoch_per_print= int(n_epoch/print_n)
    
    start_time = time.time()
    for epoch in range(n_epoch):
        #ipdb.set_trace()
        loss = tot_loss() #forward pass
        #backward propogation
        # optimizer
        lr = max(1/(epoch+10.), .001)
        optimizer = torch.optim.SGD(value.parameters(), lr, momentum = .8) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % epoch_per_print == 0:
          print('Epoch [{}/{}], Loss: {:.4f}'.format(
                  epoch+1, n_epoch, loss.item()))
        if loss.item()<0.0002:
            break
    end_time = time.time()
    print('>>>time elapsed is: ' + str(end_time - start_time))
    return value

