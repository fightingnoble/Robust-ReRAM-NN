import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union

from .quant_functions import Round_STE, IntervalQuantizeIntO, IntervalQuantize
from .quantizer import AverageLinearSignSymmIntervalQuantizer, AverageLinearSignSymmIntervalQuantizerIntO, UniformQ
round_STE = Round_STE.apply
interval_quantize_int_o = IntervalQuantizeIntO.apply
interval_quantize = IntervalQuantize.apply

minimal_num = 1e-12

from .cluster_q import insert_robust_quntizer, RobustqTorch, MiniBatchRobustqTorch

# From PIM analyser

class LayerQuantizer(nn.Module):
    def __init__(self,w_bit,a_bit,bias_bit=32,w_channel_wise=False,a_channel_wise=False) -> None:
        super().__init__()
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.bias_bit=bias_bit
        self.n_calibration_steps=1
        self.calibration_step=1
        self.calibrated=None
        self.input_process=None
        self.output_process=None
        self.weight_dynamic_process=None

        self.w_channel_wise=w_channel_wise
        self.a_channel_wise=a_channel_wise
        self.weight_interval=None
        self.input_interval=None
        self.bias_interval=None
        self.output_interval=None
        self.bias_offset=None

    def reset(self):
        self.calibrated=None
        self.weight_interval=None
        self.input_interval=None
        self.bias_interval=None
        self.output_interval=None
        self.bias_offset=None

    def quant_weight(self,weight,weight_interval=None):
        if weight_interval is None:
            weight_interval=self.weight_interval
        if weight_interval is not None:
            with torch.no_grad():
                if self.w_channel_wise:
                    weight_interval=weight_interval.reshape(-1,*[1]*(weight.dim()-1))
                max_value=2**(self.w_bit-1)
                w_int=torch.round_(weight/weight_interval).clamp(-max_value,max_value-1)
                weight=w_int*weight_interval
            # bias-correction
        return weight

    def quant_bias(self,bias,bias_interval=None):
        if bias_interval is None:
            bias_interval=self.bias_interval
        if self.bias_offset is not None:
            bias=bias+self.bias_offset
        if bias_interval is not None:
            with torch.no_grad():
                max_value=2**(self.bias_bit-1)
                bias_int=torch.round_(bias/bias_interval).clamp(-max_value,max_value-1)
                bias=bias_int*bias_interval
        return bias
    
    def quant_weight_bias(self, weight, bias):
        return self.quant_weight(weight),self.quant_bias(bias)

    def quant_features(self,tensor,interval=None):
        if interval is not None:
            if self.a_channel_wise:
                interval=interval.reshape(1,-1,*[1]*(tensor.dim()-2))
            max_value=2**(self.a_bit-1)
            a_int=torch.round_(tensor/interval).clamp_(-max_value,max_value-1)
            tensor=a_int.mul_(interval)
        return tensor
    
    def quant_input(self,tensor,input_interval=None):
        if hasattr(tensor,'interval'):
            return tensor
        if input_interval is None:
            input_interval=self.input_interval
        return self.quant_features(tensor,input_interval)
    
    def quant_output(self, tensor, output_interval=None):
        if output_interval is None:
            output_interval=self.output_interval
        return self.quant_features(tensor,output_interval)
    
    def __repr__(self) -> str:
        s=super().__repr__()
        for name in ['weight_interval','input_interval','output_interval']:
            if getattr(self,name,None) is not None:
                s+=f' {name}={getattr(self,name).view(-1)[:8].cpu().numpy()} ;'
        return s    

    @staticmethod
    def quant_calib(net,wrapped_modules,calib_loader):
        calib_layers=[]
        n_calibration_steps=1
        for name,module in wrapped_modules.items():
            module.mode='calibration_forward'
            calib_layers.append(name)
            n_calibration_steps=max(n_calibration_steps,module.quantizer.n_calibration_steps)
        print(f"prepare calibration for {calib_layers}\n n_calibration_steps={n_calibration_steps}")
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in wrapped_modules.items():
                module.quantizer.calibration_step=step+1
            with torch.no_grad():
                for inp,target in calib_loader:
                    inp=inp.cuda()
                    net(inp)
        for name,module in wrapped_modules.items():
            print(f"{name}: {module.quantizer}")
            module.mode='qat_forward'
        print("calibration finished")


class SimpleCrxbQuantizer(LayerQuantizer):
    
    def __init__(self, w_bit, a_bit, bias_bit,
                 ia_bit=None,
                 ) -> None:
        super().__init__(w_bit, a_bit, bias_bit=bias_bit)
        self.ia_bit = a_bit if ia_bit is None else ia_bit
        # activation -->> inputs quantization
        # intermediate activation -->> partial sum outputs
        # =================== Quantizer defination ===================
        self.o_quantizer = AverageLinearSignSymmIntervalQuantizer(self.ia_bit)
        self.i_quantizer = AverageLinearSignSymmIntervalQuantizerIntO(self.a_bit)
        self.w_quantizer = UniformQ(self.w_bit)

    def reset(self):
        super().reset()
        self.o_quantizer.reset()
        self.i_quantizer.reset()
        self.w_quantizer.reset()

    def quant_input(self, tensor):
        return self.i_quantizer(tensor)

    def quant_bias(self,bias):
        max_value=2**(self.bias_bit-1)
        bias_interval = bias.abs().max()/(max_value-1)
        bias_int=torch.round_(bias/bias_interval).clamp_(-max_value,max_value-1)
        bias=bias_int*bias_interval
        return bias

    def quant_weight(self, weight):
        with torch.no_grad():
            scaler = weight.abs().max() + minimal_num
        w_scale = weight / scaler
        # add clip operation
        X = torch.clamp(w_scale, -1., 1.)
        w_sim = self.w_quantizer(X)
        w_sim.interval = scaler
        return w_sim

    def quant_weight_bias(self,weight,bias):
        if bias is not None:
            b_sim = self.quant_bias(bias) 
        w_sim = self.quant_weight(weight)
        return w_sim, b_sim

    def quant_output(self,out):
        return self.o_quantizer(out)

    def calibration(self, x, weight, bias, op) -> Tensor:
        self.i_quantizer.calibration=True
        self.o_quantizer.calibration=True
        self.w_quantizer.calibration=True
        x_sim=self.quant_input(x)

        w_sim=self.quant_weight(weight)
        # self.w_quantizer.calibrated=True
        output_crxb = op(x_sim, w_sim)

        # !!! Debug 3 days!!!
        # self.w_quantizer.calibration=False
        # self.i_quantizer.calibration=False
        # self.o_quantizer.calibration=False
        # self.calibrated = True
        # return self.o_quantizer(output_crxb), x_sim, w_sim

        self.w_quantizer.calibration=False
        self.i_quantizer.calibration=False
        out_sim = self.o_quantizer(output_crxb)
        self.o_quantizer.calibration=False
        self.calibrated = True
        return out_sim, x_sim, w_sim


class RobustCrxbQuantizer(SimpleCrxbQuantizer):
    def __init__(self, w_bit, a_bit, bias_bit, alpha=0.1, gamma=1.0, ia_bit=None) -> None:
        super().__init__(w_bit, a_bit, bias_bit, ia_bit=ia_bit)
        n_clusters = 2 ** self.w_bit - 1
        self.w_quantizer = RobustqTorch(n_feature=1,
                                        n_clusters=n_clusters,
                                        alpha=alpha, gamma=gamma,
                                        n_init=1, max_iter=30, random_state=0,
                                        q_level_init="uniform"
                                        )

    def calibration(self, x, weight, bias, op) -> Tensor:
        self.i_quantizer.calibration=True
        self.o_quantizer.calibration=True
        self.w_quantizer.calibration=True
        x_sim=self.quant_input(x)

        w_sim=self.quant_weight(weight)
        self.w_quantizer.calibrated = True

        output_crxb = op(x_sim, w_sim)

        # !!! Debug 3 days!!!
        # self.w_quantizer.calibration=False
        # self.i_quantizer.calibration=False
        # self.o_quantizer.calibration=False
        # self.calibrated = True
        # return self.o_quantizer(output_crxb), x_sim, w_sim

        self.w_quantizer.calibration=False
        self.i_quantizer.calibration=False
        out_sim = self.o_quantizer(output_crxb)
        self.o_quantizer.calibration=False
        self.calibrated = True
        return out_sim, x_sim, w_sim


class MinibatchRobustCrxbQuantizer(RobustCrxbQuantizer):
    def __init__(self, w_bit, a_bit, bias_bit, alpha=0.1, gamma=1.0, ia_bit=None) -> None:
        super().__init__(w_bit, a_bit, bias_bit, ia_bit=ia_bit)
        n_clusters = 2 ** self.w_bit - 1
        self.w_quantizer = MiniBatchRobustqTorch(n_feature=1,
                                        n_clusters=n_clusters,
                                        alpha=alpha, gamma=gamma,
                                        n_init=1, max_iter=30, random_state=0,
                                        q_level_init="uniform"
                                        )


class EasyquantCrxbQuantizer(SimpleCrxbQuantizer):
    """
    Implementation of EasyQuant: Post-training Quantization via Scale Optimization arxiv2020 
    """
    def __init__(self, w_bit, a_bit, bias_bit=32, w_channel_wise=False, a_channel_wise=False, 
                 eq_alpha=0.5,eq_beta=2,eq_n=100,metric='cos_sim',input_quant=True,
                 output_quant=False,memory_constraint_GB=5,bias_error_correction=False) -> None:
        super().__init__(w_bit, a_bit, bias_bit,w_channel_wise,a_channel_wise)
        self.n_calibration_steps=3
        self.raw_outs=[]
        self.before_reorder_raw_outs=None
        assert self.a_channel_wise==False, "EasyQuant Only support layer-wise activation quantization"
        self.eq_alpha=eq_alpha
        self.eq_beta=eq_beta
        self.eq_n=eq_n
        self.metric=metric
        self.input_quant=input_quant
        self.output_quant=output_quant
        self.memory_constraint_GB=memory_constraint_GB
        self.bias_error_correction=bias_error_correction

    def use_index_weight(self, change_row_index=None):
        if self.before_reorder_raw_outs is None:
            self.before_reorder_raw_outs=[_.clone() for _ in self.raw_outs]
        with torch.no_grad():
            for i in range(len(self.raw_outs)):
                self.raw_outs[i][...]=self.before_reorder_raw_outs[i][:,change_row_index]

    def search_best_weight(self,x,weight,bias,op,raw_out,init_interval):
        if self.w_channel_wise:
            interval_candidates=[(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval for i in range(self.eq_n)]
            return channelwise_search_weight_best_quant_interval(x,op,weight,bias,raw_out,interval_candidates,self.w_bit,metric=self.metric)
        else:
            max_similarity=-1e9
            best_weight_interval=None
            best_out=None
            # print(f"Debug init_interval={init_interval}")
            # print(f"Debug x.mean()={x.mean()} x.max()={x.max()} weight.mean()={weight.mean()} weight.max()={weight.max()}")
            for i in range(self.eq_n):
                now_interval=(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval

                max_value=2**(self.w_bit-1)
                w_int=torch.round_(weight/(now_interval)).clamp(-max_value,max_value-1)

                w_sim=w_int*now_interval
                output_sim=op(x,w_sim,bias)
                # TODO: bias quantization
                if self.metric=='cos_sim':
                    similarity=F.cosine_similarity(output_sim.reshape(-1),raw_out.reshape(-1),0)
                elif self.metric=='mse':
                    similarity=-torch.mean((output_sim-raw_out)**2)
                else:
                    raise NotImplementedError()   
                # similarity=-F.mse_loss(output_sim.reshape(-1),raw_out.reshape(-1),0)
                
                if similarity>max_similarity:
                    best_weight_interval=now_interval
                    max_similarity=similarity
                    best_out=output_sim
        
        assert best_weight_interval is not None, f"similarity {similarity}"
        return best_weight_interval.detach(),best_out.detach()

    def search_best_input(self,x,weight,bias,op,raw_out,init_interval):
        if hasattr(x,'interval'):
            return x.interval,x
        else:
            assert  self.a_channel_wise==False,("No support for input channelwise quantization")
            batch_size,oc,oh,ow=raw_out.size()
            batch_size,ic,ih,iw=x.size()
            w_sim,b_sim=self.quant_weight_bias(weight,bias)
            parallel_cadidates=int((1024*1024*1024*self.memory_constraint_GB)//(raw_out.numel()*4*2+x.numel()*4))
            
            interval_candidates=(self.eq_alpha+torch.arange(self.eq_n,device=init_interval.device)/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval.view(1)
            interval_candidates=interval_candidates.view(self.eq_n,1,1,1,1)

            max_value=2**(self.a_bit-1)
            raw_out=raw_out.view(1,-1)
            similarities=[]
            for i_st in range(0,len(interval_candidates),parallel_cadidates):
                # print("parallel_cadidates",parallel_cadidates,'i_st',i_st)
                part_candidates=interval_candidates[i_st:i_st+parallel_cadidates]

                x_sim=torch.round_(x.view(1,batch_size,ic,ih,iw)/part_candidates).clamp_(-max_value,max_value-1).mul_(part_candidates) #shape #parallel batchsize ic ih iw

                out_sim=op(x_sim.view(-1,ic,ih,iw),w_sim,b_sim).view(-1,batch_size*oc*oh*ow)
                if self.metric=='cos_sim':
                    output_sim_norm=torch.norm(out_sim,1)
                    similarity=out_sim.mul_(raw_out).sum(1).div_(output_sim_norm)
                    del output_sim_norm
                elif self.metric=='mse':
                    similarity=-torch.mean(out_sim.sub_(raw_out).pow_(2))
                else:
                    raise NotImplementedError()
                del out_sim,x_sim
                similarities.append(similarity.view(-1))
            similarities=torch.cat(similarities) # shape eq_n
            max_ind=torch.argmax(similarities)
            best_input_interval=interval_candidates[max_ind].view(1)

            best_input_sim=torch.round_(x/best_input_interval).clamp_(-max_value,max_value-1).mul_(best_input_interval)
            del w_sim,b_sim,max_ind,similarities
            return best_input_interval.detach(),best_input_sim.detach()
    
    def search_best_output(self,tmp_out_sim,raw_out,init_interval):
        if self.a_channel_wise:
            interval_candidates=[(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval for i in range(self.eq_n)]
            return channelwise_search_tensor_best_quant_interval(tmp_out_sim,raw_out,interval_candidates,self.a_bit,self.metric)
        else:
            interval_candidates=[(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval for i in range(self.eq_n)]
            return search_tensor_best_quant_interval(tmp_out_sim,raw_out,interval_candidates,self.a_bit,self.metric)
    
    def get_raw_out(self,x):
        assert len(self.raw_outs)==1
        raw_out=self.raw_outs[0].to(x.device)
        # if len(self.raw_inputs)>1:
        #     raw_out=[]
        #     raw_input=[]
        #     for _raw_input,_raw_out in zip(self.raw_inputs,self.raw_outs):
        #         if _raw_input.size()==x.size():
        #             raw_out.append(_raw_out.to(x.device))
        #             raw_input.append(_raw_out.to(x.device))
        #     # raw_input=torch.cat(raw_input,0)
        #     raw_out=torch.cat(raw_out,0)
        # else:
        #     raw_out=self.raw_outs[0].to(x.device)
        #     # raw_input=self.raw_inputs[0].to(x.device)
        return raw_out
    
    def calibration(self,x,weight,bias,op):
        # step1: collection the FP32 values
        if self.calibration_step==1:
            out=op(x,weight,bias)
            self.raw_outs=[out.cpu().detach()]
            return out
        # step2: search for the best S^w of each layer
        elif self.calibration_step==2:
            # initialize
            if self.w_channel_wise:
                max=weight.data.abs().reshape(weight.size(0),-1).max(1)[0]+minimal_num
                max=max.reshape(-1,*[1]*(weight.dim()-1))
            else:
                act_max=x.data.abs().max()+minimal_num
                neg_n_intervals=1<<(self.a_bit-1)
                act_interval=act_max/(neg_n_intervals-0.5)
                x=torch.clamp_(torch.round_(x/act_interval),-neg_n_intervals,neg_n_intervals-1)*act_interval
                
                max=weight.data.abs().max()+minimal_num
            init_interval=max/(2**(self.w_bit-1)-0.5) # symmetric quantization
            raw_out=self.get_raw_out(x)
            self.weight_interval,best_out=self.search_best_weight(x,weight,bias,op,raw_out,init_interval)
            del raw_out,init_interval
            # print(f"Set weight_interval={self.weight_interval.reshape(-1)[:16]}")
            return best_out
        # step3: search for the best S^a of each layer
        elif self.calibration_step==3:
            w_sim,b_sim=self.quant_weight_bias(weight,bias)
            # initialize
            raw_out=self.get_raw_out(x)
            if self.input_quant:
                if self.a_channel_wise:
                    max=x.data.abs().amax([0,2,3])+minimal_num
                    max=max.reshape(1,-1,*[1]*(x.dim()-2))
                else:
                    max=x.data.abs().max()+minimal_num
                init_interval=max/(2**(self.a_bit-1)-0.5) # symmetric quantization
                # self.input_interval,x=self.search_best_input(x,raw_input,init_interval)
                self.input_interval,x=self.search_best_input(x,w_sim,b_sim,op,raw_out,init_interval)
                del init_interval,max
                # print(f"Set input_interval={self.input_interval.reshape(-1)[:16]}")
            out=tmp_out=op(x,w_sim,b_sim)
            if self.output_quant:
                if self.a_channel_wise:
                    max=tmp_out.data.abs().transpose(0,1).reshape(tmp_out.size(1),-1).max(1)[0]+minimal_num
                    max=max.reshape(1,-1,*[1]*(tmp_out.dim()-2))
                else:
                    max=tmp_out.data.abs().max()+minimal_num
                init_interval=max/(2**(self.a_bit-1)-0.5) # symmetric quantization
                self.output_interval,best_out=self.search_best_output(tmp_out,raw_out,init_interval)
                # print(f"Set output_interval={self.output_interval.reshape(-1)[:16]}")
                out=best_out
            if self.bias_error_correction:
                diff=raw_out-best_out
                self.bias_offset=torch.mean(diff,[0,2,3]).detach()
                out+=self.bias_offset.view(1,-1,1,1)
                del diff
            del w_sim,b_sim,raw_out
            return out


def search_tensor_best_quant_interval(tensor,raw_tensor,interval_candidates,bitwidth,metric='cos_sim', asymmetric=False, visualize=False):
    best_interval=None
    best_tensor_sim=None
    max_similarity=-1e9
    similarities = []

    if asymmetric:
        max_value=2**(bitwidth)
        offset=tensor.min()
    else:
        max_value=2**(bitwidth-1)
        offset=0

    for interval in interval_candidates:
        if asymmetric:
            tensor_q=torch.round_((tensor-offset)/interval).clamp(0,max_value-1)
        else:
            tensor_q=torch.round_((tensor-offset)/interval).clamp(-max_value,max_value-1)
        tensor_q_sim=tensor_q*interval+offset
        if metric=='cos_sim':
            similarity=F.cosine_similarity(tensor_q_sim.view(-1),raw_tensor.view(-1),0)
        elif metric=='mse':
            similarity=-F.mse_loss(tensor_q_sim.view(-1),raw_tensor.view(-1))
        else:
            raise NotImplementedError
        if visualize:
            similarities.append(similarity.cpu().detach().numpy()) # debug
        if similarity>max_similarity:
            best_interval=interval
            max_similarity=similarity
            best_tensor_sim=tensor_q_sim
    return best_interval.detach(),best_tensor_sim.detach()


def channelwise_search_tensor_best_quant_interval(tensor,raw_tensor,interval_candidates,bitwidth,metric='cos_sim'):
    """
    interval_candidates: shape [n_candidates,1,c,1,1]
    """
    max_value=2**(bitwidth-1)
    c=tensor.size(1)
    out_tensor=torch.zeros_like(raw_tensor)
    max_similarity=torch.ones(c).to(tensor.device)*-1e9
    best_tensor_intervals=torch.zeros(1,c,1,1).to(tensor.device)
    for candidate_i in range(len(interval_candidates)):
        interval=interval_candidates[candidate_i] #shape [1,c,1,1]
        # print(interval.size(),tensor.size())

        tensor_int=torch.round_(tensor/(interval)).clamp(-max_value,max_value-1)
        tensor_sim=tensor_int*interval

        if metric=='cos_sim':
            similarity=torch.mean(torch.sum(tensor_sim*raw_tensor,[2,3])/(torch.norm(tensor_sim,dim=[2,3])*torch.norm(raw_tensor,dim=[2,3])),0) # shape c
        elif metric=='mse':
            similarity=-torch.mean((tensor_sim-raw_tensor)**2,[0,2,3])
        else:
            raise NotImplementedError()    
        mask=(similarity>max_similarity).view(1,-1,1,1).float()
        best_tensor_intervals=mask*interval+(1-mask)*best_tensor_intervals
        # best_weight_intervals.masked_fill_(mask.view(-1,1,1,1),interval)
        mask=mask.view(-1)
        max_similarity=mask*similarity+(1-mask)*max_similarity
        # max_similarity.masked_fill_(mask,similarity)
        mask=mask.view(1,-1,1,1)
        out_tensor=mask*tensor_sim+(1-mask)*out_tensor
        # out_tensor.masked_fill_(mask.view(1,-1,1,1),out_sim)
    # print(interval_candidates[0].view(-1),interval_candidates[-1].view(-1),best_tensor_intervals.view(-1))
    return best_tensor_intervals.detach(),out_tensor.detach()


def channelwise_search_weight_best_quant_interval(input,op,weight,bias,raw_output,interval_candidates,bitwidth,metric='cos_sim'):
    """
    interval_candidates: shape [n_candidates,oc,1,1,1]
    """
    with torch.no_grad():
        max_value=2**(bitwidth-1)
        oc=weight.size(0)
        out_tensor=torch.zeros_like(raw_output)
        max_similarity=torch.ones(oc).to(input.device)*-1e9
        best_weight_intervals=torch.zeros(oc,1,1,1).to(input.device)
        for candidate_i in range(len(interval_candidates)):
            interval=interval_candidates[candidate_i]

            w_int=torch.round_(weight/(interval)).clamp(-max_value,max_value-1)
            w_sim=w_int*interval
            out_sim=op(input,w_sim,bias)
            if metric=='cos_sim':
                similarity=torch.mean(torch.sum(out_sim*raw_output,[2,3])/(torch.norm(out_sim,dim=[2,3])*torch.norm(raw_output,dim=[2,3])),0) # shape c
            elif metric=='mse':
                similarity=-torch.mean((out_sim-raw_output)**2,[0,2,3])
            else:
                raise NotImplementedError()
            mask=(similarity>max_similarity).view(-1,1,1,1).float()
            best_weight_intervals=mask*interval+(1-mask)*best_weight_intervals
            # best_weight_intervals.masked_fill_(mask.view(-1,1,1,1),interval)
            mask=mask.view(-1)
            max_similarity=mask*similarity+(1-mask)*max_similarity
            # max_similarity.masked_fill_(mask,similarity)
            mask=mask.view(1,-1,1,1)
            out_tensor=mask*out_sim+(1-mask)*out_tensor
            # out_tensor.masked_fill_(mask.view(1,-1,1,1),out_sim)
    return best_weight_intervals.detach(),out_tensor.detach()
        