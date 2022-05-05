from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from .gru_layernorm_cell import LayerNormGRUCell
from .layers import *

NDIM_HIDDEN = 64


class MultiGATConvGRU(nn.Module):
    def __init__(
        self,
        nin1:int,
        nin2:int,
        nout1:int,
        nout2:int,
        total_factor_types:Optional[Union[Tuple,List]]=None,
        nhid:int=64,
        nstate:int=64,
        nstep:int=0,
        nlayer:int = 1,  # number of GAT layers
        heads:int=3,
        n_hidden=1, # number of MLP hidden layers used for message
        rnn_method:str="gru",
        const_factor_input:bool=True,
        init_method:str="encode",
        decode_method:str="varstate_mlp",
        gat_module:str = "gat",
        aggregation_method:str="sum",
        asrnn_kwargs:dict={},
        damping:float=0.0,  # smoothing the hidden state update
        decode_series_len:int=1,  # used with decode_method='varseries_mlp'
        add_self_loops=False,  #whether to add self loop for the attention, since we are using bipartite graph, default to False
        use_factor_net:bool=False,   # map factor parameter to a matrix and multiply with aggregated message
        use_bn=False,  #wether to use batch norm in MLP
        use_att=True, #whether to use attention or set them to constant 1
        message_nonlinearity=None, # nonlinearity applied to messages
        **kwargs,
    ):
        super(MultiGATConvGRU, self).__init__()
        self.nin1 = nin1
        self.nin2 = nin2
        self.nout1 = nout1
        self.nout2 = nout2
        self.nhid = nhid
        self.nstate = nstate
        self.nstep = nstep
        self.nlayer = nlayer
        self.n_hidden = n_hidden
        self.rnn_method = rnn_method
        self.aggregation_method = aggregation_method
        self.heads = heads
        self.num_factor_types = len(total_factor_types)
        self.init_method = init_method
        self.decode_method = decode_method
        self.decode_series_len = decode_series_len
        self.damping = damping
        self.const_factor_input = const_factor_input
        self.add_self_loops = add_self_loops
        self.use_bn = use_bn
        self.use_att = use_att
        if message_nonlinearity is None:
            self.message_nonlinearity = lambda x:x
        elif message_nonlinearity == "elu":
            self.message_nonlinearity = torch.nn.ELU()
        elif message_nonlinearity == "selu":
            self.message_nonlinearity = torch.nn.SELU()
        elif message_nonlinearity == "sigmoid":
            self.message_nonlinearity = torch.nn.Sigmoid()
        else:
            raise NotImplementedError(f"nonlinearity {message_nonlinearity} not implememted")


        # Construct encoders and decoders
        encoder1_list = encoder2 = decoder1_list = decoder2 = None
        encoder1_list = [
            MLP(nin1, nhid, nstate, n_hidden=2, use_bn=use_bn) for _ in range(len(total_factor_types))
        ]
        self.encoder1_list = (
            torch.nn.ModuleList(
                encoder1_list) if encoder1_list is not None else None
        )
        encoder2 = MLP(nin2, nhid, nstate, n_hidden=2, use_bn=use_bn)
        self.encoder2 = encoder2

        if nout1 is not None:
            decoder1_list = [
                MLP(nstate, nhid, nout1, n_hidden=2, use_bn=use_bn)
                for _ in range(len(total_factor_types))
            ]
            self.decoder1_list = (
                torch.nn.ModuleList(decoder1_list)
                if decoder1_list is not None
                else None
            )
        self.decoder1 = self.decoder2 = None
        if decode_method == "varstate_mlp":
            decoder2 = MLP(nstate, nhid, nout2, n_hidden=2, use_bn=use_bn)
            self.decoder2 = decoder2
        elif decode_method == "factorstate_mlp":
            decoder1 = MLP(nstate, nhid, nout1, n_hidden=2, use_bn=use_bn)
            self.decoder1 = decoder1
        elif decode_method == "varseries_mlp":
            decoder2 = MLP(nstate*decode_series_len,
                            nhid, nout2, n_hidden=2, use_bn=use_bn)
            self.decoder2 = decoder2
        elif decode_method == "varseriesmean_mlp":
            decoder2 = MLP(nstate, nhid, nout2, n_hidden=2, use_bn=use_bn)
            self.decoder2 = decoder2
        elif decode_method == "varseriesmlp_mean":
            decoder2 = MLP(nstate, nhid, nout2, n_hidden=2, use_bn=use_bn)
            self.decoder2 = decoder2
        elif decode_method == "varstate_linear":
            decoder2 = torch.nn.Linear(nstate, nout2, bias=True)
            self.decoder2 = decoder2
        elif decode_method == "varseries_linear":
            decoder2 = torch.nn.Linear(
                nstate*decode_series_len, nout2, bias=True)
            self.decoder2 = decoder2
        elif decode_method == "varseriesmean_linear":
            decoder2 = torch.nn.Linear(nstate, nout2, bias=True)
            self.decoder2 = decoder2
        elif decode_method == "varserieslinear_mean":
            decoder2 = torch.nn.Linear(nstate, nout2, bias=True)
            self.decoder2 = decoder2
        else:
            raise NotImplementedError("decode method not implemented")

        self.register_buffer(
            "ftypes", torch.tensor(total_factor_types, dtype=torch.long)
        )
        if self.aggregation_method in ["mean"]:
            nstate_message = nstate
            # nstate_message_pre = nstate
        elif self.aggregation_method in ["sum"]:
            nstate_message = nstate
            # nstate_message_pre = nstate *  heads
        elif self.aggregation_method in ["cat"]:
            nstate_message = nstate * heads
            # nstate_message_pre = nstate * heads
        else:
            raise NotImplementedError("aggregation method not implemented")
        if self.aggregation_method in ["sum", "cat"]:
            concat = True
        else:
            concat = False
        self.nstate_message = nstate_message

        ## Propagator model
        self.gat_module = gat_module
        if gat_module == 'mygat':
            GATModule = MyGATConv
            gat_kwargs = {'n_hidden': n_hidden, 'hid_channels': nhid}
        elif gat_module == 'gat':
            GATModule = GATConv
            gat_kwargs = {}
        elif gat_module == 'gatv2':
            GATModule = GATv2Conv
            gat_kwargs = {}
        else:
            raise NotImplementedError(f"GAT module name {gat_module} not supported")
        #  1 --> 2
        if self.const_factor_input:
            self.propagator12 = GATModule( (2*nstate,nstate), nstate, heads=self.heads, concat=concat, add_self_loops=self.add_self_loops, **gat_kwargs)  
        else:
            self.propagator12 = GATModule( nstate, nstate, heads=self.heads, concat=concat, add_self_loops=self.add_self_loops, **gat_kwargs)  

        self.propagator21_list = nn.ModuleList(
            [
                GATModule(nstate, nstate, heads=self.heads,
                        concat=concat, add_self_loops=self.add_self_loops, **gat_kwargs)
                for _ in range(self.num_factor_types)
            ]
        )
        if not self.use_att:
            gat = self.propagator12
            if hasattr(gat,'att_l'):
                gat.att_l.requires_grad = False
                gat.att_r.requires_grad = False
                gat.att_l.fill_(0.0)
                gat.att_r.fill_(0.0)
                for gat in self.propagator21_list:
                    gat.att_l.requires_grad = False
                    gat.att_r.requires_grad = False
                    gat.att_l.fill_(0.0)
                    gat.att_r.fill_(0.0)
            else:
                gat.att_src.requires_grad = False
                gat.att_dst.requires_grad = False
                gat.att_src.fill_(0.0)
                gat.att_dst.fill_(0.0)
                for gat in self.propagator21_list:
                    gat.att_src.requires_grad = False
                    gat.att_dst.requires_grad = False
                    gat.att_src.fill_(0.0)
                    gat.att_dst.fill_(0.0)
        
        self.use_factor_net = use_factor_net
        if self.use_factor_net: 
            self.factor_net = nn.Linear(nstate, nstate_message *nstate_message,bias=True)

        if self.rnn_method == 'gru':
            self.rnn1_list = nn.ModuleList(
                [
                    nn.GRUCell(nstate_message, nstate, bias=True)
                    for _ in range(self.num_factor_types)
                ]
            )
            self.rnn2 = nn.GRUCell(nstate_message, nstate, bias=True)

        elif self.rnn_method == 'gru_ln':  
            self.rnn1_list = nn.ModuleList(
                [
                    LayerNormGRUCell(nstate_message, nstate, bias=True)
                    for _ in range(self.num_factor_types)
                ]
            )
            self.rnn2 = LayerNormGRUCell(nstate_message, nstate, bias=True)

        elif self.rnn_method == 'asrnn':  # Antisymmetric RNN
            self.rnn1_list = nn.ModuleList(
                [
                    AntisymmetricRNN(nstate_message, nstate, **asrnn_kwargs)
                    for _ in range(self.num_factor_types)
                ]
            )
            self.rnn2 = AntisymmetricRNN(
                nstate_message, nstate, **asrnn_kwargs)
        elif self.rnn_method == 'myasrnn':  # My asrnn
            self.rnn1_list = nn.ModuleList(
                [
                    MyAntisymmetricRNN(nstate_message, nstate)
                    for _ in range(self.num_factor_types)
                ]
            )
            self.rnn2 = MyAntisymmetricRNN(nstate_message, nstate)
        elif self.rnn_method == 'cfn':  # Chaos Free Network
            self.rnn1_list = nn.ModuleList(
                [
                    CFN(nstate_message, nstate)
                    for _ in range(self.num_factor_types)
                ]
            )
            self.rnn2 = CFN(nstate_message, nstate)
        elif self.rnn_method == 'minimalrnn':  # Chaos Free Network
            self.rnn1_list = nn.ModuleList(
                [
                    MinimalRNN(nstate_message, nstate)
                    for _ in range(self.num_factor_types)
                ]
            )
            self.rnn2 = MinimalRNN(nstate_message, nstate)
        elif self.rnn_method == "none":
            pass
        else:
            raise NotImplementedError("Given RNN method not implemented")
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.decoder2 is not None:
            nn.init.normal_(self.decoder2.output.weight.data, std=1e-8)
            nn.init.zeros_(self.decoder2.output.weight.data)
        if self.decoder1 is not None:
            nn.init.normal_(self.decoder1.output.weight.data, std=1e-8)
            nn.init.zeros_(self.decoder1.output.weight.data)

    def one_step(self, g:Data, h1:Tensor, h2:Tensor, masks:Optional[List[Tensor]]=None, 
                    f1:Optional[Tensor]=None,  # initial factor embedding for constant input
                    factor_matrix:Optional[Tensor]=None,  
                    return_attention_weights:bool=True,
                    calculate_factor_matrix:bool=True,
                    )->Tuple[Tensor]:
        """One step of RNN computation

        Args:
            g (Graph): the input graph
            h1 (Tensor): latent state for factor nodes
            h2 (Tensor): latent state for variable nodes

        Returns:
            new_h1
            new_h2 
            m1
            m2
        """
        if calculate_factor_matrix:
            factor_matrix = self._get_factor_matrix(g)
        def pool_message_post(message):
            if self.aggregation_method in ["mean", "cat"]:
                return message
            elif self.aggregation_method == "sum":
                message = message.view(
                    *message.shape[:-1], self.nstate, self.heads).sum(-1)
                return message
        if masks is None:
            factor_types = g.x1[:, 0].long()
            masks = [(factor_types == self.ftypes[i])
                     for i in range(self.num_factor_types)]
        size1 = g.x1.shape[0]
        size2 = g.x2.shape[0]
        if self.const_factor_input:
            h1_extended = torch.cat([h1,f1], dim=-1)
        else:
            h1_extended = h1
        m2, (_, alpha2) = self.propagator12.forward(
            (h1_extended, h2), g.edge_index, return_attention_weights=return_attention_weights
        )
        m2 = self.message_nonlinearity(m2)
        m2 = pool_message_post(m2)
        m1 = torch.zeros(*h1.shape[:-1], self.nstate_message, device=h1.device)
        new_h1 = torch.zeros_like(h1)
        for i in range(self.num_factor_types):
            tmp, (_, alpha1) = self.propagator21_list[i].forward(
                (h2, h1),
                torch.stack([g.edge_index[1], g.edge_index[0]]),
                 return_attention_weights=return_attention_weights
            )
            tmp = pool_message_post(tmp)
            tmp = self.message_nonlinearity(tmp)
            if self.use_factor_net:
                tmp = (factor_matrix @ tmp.unsqueeze(-1)).squeeze(-1)
            m1[masks[i]] = tmp[masks[i]]

        if self.rnn_method != "none":
            new_h2 = self.rnn2(m2, h2) * (1.0-self.damping) + h2 * self.damping
            for i in range(self.num_factor_types):
                new_h1[masks[i]] = self.rnn1_list[i](
                    m1[masks[i]], h1[masks[i]]
                ) * (1.0-self.damping) + h1[masks[i]] * self.damping
        else:
            new_h2 = m2 * (1.0-self.damping) + h2 * self.damping
            new_h1 = m1 * (1.0-self.damping) + h1 * self.damping

        if return_attention_weights:
            return new_h1, new_h2, m1, m2, alpha1, alpha2
        else:
            return new_h1, new_h2, m1, m2

    def _get_f1(self, g:Data):
        factor_types = g.x1[:, 0].long()
        model_device = g.x1.device
        masks = [(factor_types == self.ftypes[i])
                 for i in range(self.num_factor_types)]
        f1 = torch.zeros(g.x1.shape[0], self.nstate, device=model_device)           # representation of factor parameters
        for i in range(self.num_factor_types):
            f1[masks[i]] = self.encoder1_list[i](g.x1[masks[i]])
        return f1

    def _get_factor_matrix(self, g:Data):
        f1 = self._get_f1(g)
        factor_matrix = self.factor_net(f1).view(-1, self.nstate_message, self.nstate_message)
        return factor_matrix
        
    def forward(self, g:Data, nstep:Optional[int]=None, output_dynamics:bool=False)->Union[Tensor,Tuple]:
        """
        g should have following fields:
        x1: feature of first class of nodes
        x2: features of second class of nodes
        y1: features of target on first class, can be None
        y2: features of target on second class, can be None
        edge_index: shape (2,num_edge), from first class to second class

        Args
        - g: the input graph
        - nstep: number recurrent steps to run
        - output_dynamics: return latent states dynamics along with final readout
        """
        model_device = g.x1.device
        # model_device = next(self.parameters()).device
        size1 = g.x1.shape[0]
        size2 = g.x2.shape[0]
        factor_types = g.x1[:, 0].long()
        masks = [(factor_types == self.ftypes[i])
                 for i in range(self.num_factor_types)]
        assert (
            sum([sum(m.long()).item() for m in masks]) == g.x1.shape[0]
        ), "size incorrect {}:{}".format(sum([len(m) for m in masks]), g.x1.shape[0])
        if nstep is None:
            nstep = self.nstep

        # Initialize
        h1 = torch.zeros(g.x1.shape[0], self.nstate, device=model_device)
        h2 = torch.zeros(g.x2.shape[0], self.nstate, device=model_device)
        f1 = self._get_f1(g)
        if self.init_method == "encode":
            for i in range(self.num_factor_types):
                h1[masks[i]] = self.encoder1_list[i](g.x1[masks[i]])
            h2 = self.encoder2(g.x2)
        elif self.init_method == "randn":
            torch.nn.init.normal_(h1)
            torch.nn.init.normal_(h2)
        elif self.init_method == "zero":
            pass
        else:
            raise NotImplementedError("init method not implemented")
        if self.use_factor_net:
            factor_matrix = self.factor_net(f1).view(-1, self.nstate_message, self.nstate_message)
        else:
            factor_matrix = None
        h1list = [h1] + [None for _ in range(nstep)]
        h2list = [h2] + [None for _ in range(nstep)]
        alpha1list  = [None for _ in range(nstep+1)]
        alpha2list  = [None for _ in range(nstep+1)]
        m1list = [torch.zeros(*h1.shape[:-1], self.nstate_message)
                  ] + [None for _ in range(nstep)]
        m2list = [torch.zeros(*h2.shape[:-1], self.nstate_message)
                  ] + [None for _ in range(nstep)]

        for t in range(1, nstep + 1):
            h1list[t], h2list[t], m1list[t], m2list[t], alpha1list[t], alpha2list[t] = self.one_step( g, h1list[t-1], h2list[t - 1], masks=masks, return_attention_weights=True, f1=f1, factor_matrix=factor_matrix, calculate_factor_matrix=False)
        if nstep>0:
            alpha1list[0] = torch.zeros_like(alpha1list[-1])    # so can be stacked later
            alpha2list[0] = torch.zeros_like(alpha2list[-1])

        # Decoding
        yhat = None
        if self.decode_method in ["varstate_mlp", "varstate_linear"]:
            yhat = self.decoder2(h2list[-1])
        elif self.decode_method in ["factorstate_mlp", "factorstate_linear"]:
            yhat = self.decoder1(h1list[-1])
        elif self.decode_method in ["varseries_mlp", "varseries_linear"]:
            yhat = self.decoder2(
                torch.cat(h2list[-self.decode_series_len:], dim=-1)
            )
        elif self.decode_method in ["varseriesmean_mlp", 'varseriesmean_linear']:
            yhat = self.decoder2(
                torch.mean(
                    torch.stack(h2list[-self.decode_series_len:], dim=-1), dim=-1
                ).squeeze()
            )
        elif self.decode_method in ["varseriesmlp_mean", "varserieslinear_mean"]:
            inp = torch.stack(h2list[-self.decode_series_len:], dim=0)
            yhat = self.decoder2(inp).mean(dim=0)
            # yhat2list = [self.decoder2(h2list[-i]) for i in range(1,1+self.decode_series_len)]

        else:
            raise NotImplementedError(
                f"Decode method {self.decode_method} not implemented. And should be checkecd earlier")
        
        if not output_dynamics:
            return yhat
        else:
            return (yhat, [
                h1list, h2list, m1list, m2list, alpha1list, alpha2list
            ])



class MultiGATConvStack(nn.Module):
    """
    Stacked Factor-GNN
    """

    def __init__(
            self,
            nin1: int,
            nin2: int,
            nout1: int,
            nout2: int,
            total_factor_types: Optional[Union[Tuple, List]] = None,
            nhid: int = 64,
            nstate: int = 64,
            nstep: int = 0,
            heads: int = 3,
            nlayer: int = 5,  # number of stacked GNN layers
            const_factor_input: bool = True,
            init_method: str = "encode",
            decode_method: str = "varstate_mlp",
            aggregation_method: str = "sum",
            damping: float = 0.0,  # smoothing the hidden state update
            decode_series_len: int = 1,  # used with decode_method='varseries_mlp'
            add_self_loops=False,
            # whether to add self loop for the attention, since we are using bipartite graph, default to False
            use_factor_net: bool = False,  # map factor parameter to a matrix and multiply with aggregated message
            use_bn=False,  # wether to use batch norm in MLP
            **kwargs,
    ):
        super(MultiGATConvStack, self).__init__()
        self.nin1 = nin1
        self.nin2 = nin2
        self.nout1 = nout1
        self.nout2 = nout2
        self.nhid = nhid
        self.nstate = nstate
        self.nstep = nstep
        self.nlayer = nlayer
        self.aggregation_method = aggregation_method
        self.heads = heads
        self.num_factor_types = len(total_factor_types)
        self.init_method = init_method
        self.decode_method = decode_method
        self.decode_series_len = decode_series_len
        self.damping = damping
        self.const_factor_input = const_factor_input
        self.add_self_loops = add_self_loops
        self.use_bn = use_bn

        # Construct encoders and decoders
        encoder1_list = [
            MLP(nin1, nhid, nstate, n_hidden=2, use_bn=use_bn) for _ in range(len(total_factor_types))
        ]
        self.encoder1_list = (
            torch.nn.ModuleList(
                encoder1_list) if encoder1_list is not None else None
        )
        encoder2 = MLP(nin2, nhid, nstate, n_hidden=2, use_bn=use_bn)
        self.encoder2 = encoder2

        if nout1 is not None:
            decoder1_list = [
                MLP(nstate, nhid, nout1, n_hidden=2, use_bn=use_bn)
                for _ in range(len(total_factor_types))
            ]
            self.decoder1_list = (
                torch.nn.ModuleList(decoder1_list)
                if decoder1_list is not None
                else None
            )
        if nout2 is not None:
            if decode_method=="varstate_mlp":
                decoder2 = MLP(nstate, nhid, nout2, n_hidden=2, use_bn=use_bn)
                self.decoder2 = decoder2
            elif decode_method=="varseries_mlp":
                decoder2 = MLP(nstate*decode_series_len,
                               nhid, nout2, n_hidden=2, use_bn=use_bn)
                self.decoder2 = decoder2
            else:
                raise NotImplementedError("decode method not implemented")

        self.register_buffer(
            "ftypes", torch.tensor(total_factor_types, dtype=torch.long)
        )
        if self.aggregation_method in ["mean"]:
            nstate_message = nstate
        elif self.aggregation_method in ["sum"]:
            nstate_message = nstate
        elif self.aggregation_method in ["cat"]:
            nstate_message = nstate*heads
        else:
            raise NotImplementedError("aggregation method not implemented")
        if self.aggregation_method in ["sum", "cat"]:
            concat = True
        else:
            concat = False
        self.nstate_message = nstate_message

        #  1 --> 2
        if self.const_factor_input:
            self.propagator12_list = nn.ModuleList([
                GATConv((2*nstate, nstate), nstate, heads=self.heads, concat=concat, add_self_loops=self.add_self_loops)
                for _ in range(nlayer)
            ])
        else:
            self.propagator12_list = nn.ModuleList([
                GATConv(nstate, nstate, heads=self.heads, concat=concat, add_self_loops=self.add_self_loops)
                for _ in range(nlayer)
            ])

        self.propagator21_list = nn.ModuleList([
            nn.ModuleList([
                GATConv(nstate, nstate, heads=self.heads,
                        concat=concat, add_self_loops=self.add_self_loops)
                for _ in range(self.nlayer)])
            for _ in range(self.num_factor_types)
        ])

        self.use_factor_net = use_factor_net
        if self.use_factor_net:
            self.factor_net = nn.Linear(nstate, nstate_message*nstate_message, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.decoder2.output.weight.data, std=1e-8)
        nn.init.zeros_(self.decoder2.output.weight.data)

    def one_step(self, ilayer: int, g: Data, h1: Tensor, h2: Tensor, masks: Optional[List[Tensor]] = None,
                 f1: Optional[Tensor] = None,  # initial factor embedding for constant input
                 factor_matrix: Optional[Tensor] = None,
                 return_attention_weights: bool = True,
                 calculate_factor_matrix: bool = True,
                 ) -> Tuple[Tensor]:
        if calculate_factor_matrix:
            factor_matrix = self._get_factor_matrix(g)

        def pool_message_post(message):
            if self.aggregation_method in ["mean", "cat"]:
                return message
            elif self.aggregation_method=="sum":
                message = message.view(
                    *message.shape[:-1], self.nstate, self.heads).sum(-1)
                return message

        if masks is None:
            factor_types = g.x1[:, 0].long()
            masks = [(factor_types==self.ftypes[i])
                     for i in range(self.num_factor_types)]
        size1 = g.x1.shape[0]
        size2 = g.x2.shape[0]
        if self.const_factor_input:
            h1_extended = torch.cat([h1, f1], dim=-1)
        else:
            h1_extended = h1
        m2, (_, alpha2) = self.propagator12_list[ilayer].forward(
            (h1_extended, h2), g.edge_index, size=(size1, size2), return_attention_weights=return_attention_weights
        )
        m2 = pool_message_post(m2)
        m1 = torch.zeros(*h1.shape[:-1], self.nstate_message, device=h1.device)
        new_h1 = torch.zeros_like(h1)
        for i in range(self.num_factor_types):
            tmp, (_, alpha1) = self.propagator21_list[i][ilayer].forward(
                (h2, h1),
                torch.stack([g.edge_index[1], g.edge_index[0]]),
                size=(size2, size1), return_attention_weights=return_attention_weights
            )
            tmp = pool_message_post(tmp)
            if self.use_factor_net:
                tmp = (factor_matrix@tmp.unsqueeze(-1)).squeeze(-1)
            m1[masks[i]] = tmp[masks[i]]

        new_h2 = m2*(1.0 - self.damping) + h2*self.damping
        new_h1 = m1*(1.0 - self.damping) + h1*self.damping

        if return_attention_weights:
            return new_h1, new_h2, m1, m2, alpha1, alpha2
        else:
            return new_h1, new_h2, m1, m2

    def _get_f1(self, g: Data):
        factor_types = g.x1[:, 0].long()
        model_device = g.x1.device
        masks = [(factor_types==self.ftypes[i])
                 for i in range(self.num_factor_types)]
        f1 = torch.zeros(g.x1.shape[0], self.nstate, device=model_device)  # representation of factor parameters
        for i in range(self.num_factor_types):
            f1[masks[i]] = self.encoder1_list[i](g.x1[masks[i]])
        return f1

    def _get_factor_matrix(self, g: Data):
        f1 = self._get_f1(g)
        factor_matrix = self.factor_net(f1).view(-1, self.nstate_message, self.nstate_message)
        return factor_matrix

    def forward(self, g: Data, nstep: Optional[int] = None, output_dynamics: bool = False) -> Union[Tensor, Tuple]:
        """
        g should have following fields:
        x1: feature of first class of nodes
        x2: features of second class of nodes
        y1: features of target on first class, can be None
        y2: features of target on second class, can be None
        edge_index: shape (2,num_edge), from first class to second class

        Args
        - g: the input graph
        - nstep: number recurrent steps to run
        - output_dynamics: return latent states dynamics along with final readout
        """
        model_device = g.x1.device
        # model_device = next(self.parameters()).device
        size1 = g.x1.shape[0]
        size2 = g.x2.shape[0]
        factor_types = g.x1[:, 0].long()
        masks = [(factor_types==self.ftypes[i])
                 for i in range(self.num_factor_types)]
        assert (
                sum([sum(m.long()).item() for m in masks])==g.x1.shape[0]
        ), "size incorrect {}:{}".format(sum([len(m) for m in masks]), g.x1.shape[0])
        if nstep is None:
            nstep = self.nstep

        # Initialize
        h1 = torch.zeros(g.x1.shape[0], self.nstate, device=model_device)
        h2 = torch.zeros(g.x2.shape[0], self.nstate, device=model_device)
        f1 = self._get_f1(g)
        if self.init_method=="encode":
            for i in range(self.num_factor_types):
                h1[masks[i]] = self.encoder1_list[i](g.x1[masks[i]])
            h2 = self.encoder2(g.x2)
        elif self.init_method=="randn":
            torch.nn.init.normal_(h1)
            torch.nn.init.normal_(h2)
        elif self.init_method=="zero":
            pass
        else:
            raise NotImplementedError("init method not implemented")
        if self.use_factor_net:
            factor_matrix = self.factor_net(f1).view(-1, self.nstate_message, self.nstate_message)
        else:
            factor_matrix = None
        h1list = [h1] + [None for _ in range(self.nlayer)]
        h2list = [h2] + [None for _ in range(self.nlayer)]
        alpha1list = [None for _ in range(self.nlayer + 1)]
        alpha2list = [None for _ in range(self.nlayer + 1)]
        m1list = [torch.zeros(*h1.shape[:-1], self.nstate_message)
                  ] + [None for _ in range(self.nlayer)]
        m2list = [torch.zeros(*h2.shape[:-1], self.nstate_message)
                  ] + [None for _ in range(self.nlayer)]

        for t in range(1, self.nlayer + 1):
            h1list[t], h2list[t], m1list[t], m2list[t], alpha1list[t], alpha2list[t] = self.one_step(t - 1, g,
                                                                                                     h1list[t - 1],
                                                                                                     h2list[t - 1],
                                                                                                     masks=masks,
                                                                                                     return_attention_weights=True,
                                                                                                     f1=f1,
                                                                                                     factor_matrix=factor_matrix,
                                                                                                     calculate_factor_matrix=False)
        if self.nlayer > 0:
            alpha1list[0] = torch.zeros_like(alpha1list[-1])  # so can be stacked later
            alpha2list[0] = torch.zeros_like(alpha2list[-1])

        # Decoding
        yhat1 = yhat2 = None
        if self.decode_method in ["varstate_mlp", "varstate_linear"]:
            yhat2 = self.decoder2(h2list[-1])
        elif self.decode_method in ["varseries_mlp", "varseries_linear"]:
            yhat2 = self.decoder2(
                torch.cat(h2list[-self.decode_series_len:], dim=-1)
            )
        elif self.decode_method in ["varseriesmean_mlp", 'varseriesmean_linear']:
            yhat2 = self.decoder2(
                torch.mean(
                    torch.stack(h2list[-self.decode_series_len:], dim=-1), dim=-1
                ).squeeze()
            )
        elif self.decode_method in ["varseriesmlp_mean", "varserieslinear_mean"]:
            inp = torch.stack(h2list[-self.decode_series_len:], dim=0)
            yhat2 = self.decoder2(inp).mean(dim=0)
            # yhat2list = [self.decoder2(h2list[-i]) for i in range(1,1+self.decode_series_len)]

        else:
            raise NotImplementedError(
                "Decode method not implemented. And should be checkecd earlier")
        if not output_dynamics:
            return yhat2
        else:
            return (yhat2, [
                h1list, h2list, m1list, m2list, alpha1list, alpha2list
            ])
