import torch
import numpy as np
from torch.autograd import Variable

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class HyperOptimizer(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.hyper_parameters(),
        lr=args.lw_learning_rate, betas=(0.5, 0.999), weight_decay=args.lw_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, vgg_model,loss_fn_alex,network_optimizer):
    output=self.model(input)
    loss = self.model.search_tr_loss(output, target,vgg_model,loss_fn_alex)[0]
    theta = _concat(self.model.net_parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.net_parameters()).mul_(self.network_momentum)#parameters
    except:
      moment = torch.zeros_like(theta)
    grad_=torch.autograd.grad(loss, self.model.net_parameters())
    dtheta = _concat(grad_).data + self.network_weight_decay*theta#parameters
    unrolled_model = self._construct_model_from_theta(theta.sub(eta[0]*(moment + dtheta)))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, vgg_model,loss_fn_alex,classifier,network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        val_loss=self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta,vgg_model,loss_fn_alex,classifier, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid,classifier)
    self.optimizer.step()
    return val_loss

  def _backward_step(self, input_valid, target_valid,classifier):
    output_valid=self.model(input_valid)
    loss  = self.model.search_val_loss(output_valid, classifier)[0]
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, vgg_model,loss_fn_alex,classifier,network_optimizer):
    # caculate formulation: dβLval(w',β) ,where w' = w − ξ*dwLtrain(w, β)
    # w'
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, vgg_model,loss_fn_alex,network_optimizer)
    # Lval
    output_valid=unrolled_model(input_valid)
    # caculate Lval use optimized network unrolled_model by w
    unrolled_loss = unrolled_model.search_val_loss(output_valid , classifier)[0]
    unrolled_loss.backward()
    # dβLval(w',β) β
    dbeta = []
    for v in unrolled_model.hyper_parameters():
      if v.grad is None:
        dbeta.append(torch.zeros_like(v))
      else:
        dbeta.append(v.grad)
    vector = [v.grad.data for v in unrolled_model.net_parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train,vgg_model,loss_fn_alex)
    # fomulation 6-8
    for g, ig in zip(dbeta, implicit_grads):
      g.data.sub_(eta[0]*ig.data)

    # optimize beta
    for v, g in zip(self.model.hyper_parameters(), dbeta):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    return unrolled_loss.item()

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new_hyper()
    params, offset = {}, 0
    for v1, v2 in zip(model_new.net_parameters(),self.model.net_parameters()):
      v_length = np.prod(v2.size())
      v1.data.copy_(theta[offset: offset+v_length].view(v2.size()))
      offset += v_length
    assert offset == len(theta)
    return model_new.cuda()
  def _hessian_vector_product(self, vector, input, target,vgg_model,loss_fn_alex, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.net_parameters(), vector):
      p.data.add_(R, v)

    output = self.model(input)
    loss = self.model.search_tr_loss(output, target,vgg_model,loss_fn_alex)[0]

    grads_p = torch.autograd.grad(loss, self.model.hyper_parameters())
    for p, v in zip(self.model.net_parameters(), vector):
      p.data.sub_(2*R, v)

    output = self.model(input)
    loss = self.model.search_tr_loss(output, target, vgg_model, loss_fn_alex)[0]
    grads_n = torch.autograd.grad(loss, self.model.hyper_parameters())
    for p, v in zip(self.model.net_parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

