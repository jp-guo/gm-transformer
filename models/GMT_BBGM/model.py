import copy
import torch
import itertools
from torch import nn

from models.GMT_BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.GMT_BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from lpmp_py import GraphMatchingModule
from lpmp_py import MultiGraphMatchingModule
from src.feature_align import feature_align

from src.utils.config import cfg
from src.backbone import *
GMT = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Net(GMT):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.BBGM.FEATURE_CHANNEL)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.BBGM.FEATURE_CHANNEL
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        num_graphs = len(images)

        if cfg.PROBLEM.TYPE == '2GM' and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        elif cfg.PROBLEM.TYPE == 'MGM' and 'gt_perm_mat' in data_dict:
            perm_mat_list = data_dict['gt_perm_mat']
            gt_perm_mats = [torch.bmm(pm_src, pm_tgt.transpose(1, 2)) for pm_src, pm_tgt in lexico_iter(perm_mat_list)]
        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []
        old_shape = data_dict['gt_perm_mat'][0].shape
        max_shape = max(data_dict['gt_perm_mat'][0].shape)
        perm_old = torch.zeros([len(data_dict['gt_perm_mat']), max_shape, max_shape],
                               device=data_dict['gt_perm_mat'].device)
        perm_old[:, 0:old_shape[0], 0:old_shape[1]] = data_dict['gt_perm_mat']
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes, edges, glb = self.gmt(image, p, n_p)

            # global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))

            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            global_list.append(glb)

            # arrange features
            U = concat_features(nodes, n_p)
            F = concat_features(edges, n_p)
            node_features = torch.cat((U, F), dim=1)

            graph.x = node_features
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)

            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_affs_list]

        if self.training:
            unary_costs_list = [
                [
                    x + 1.0 * gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, gt_perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, gt_perm_mat, (ns_src, ns_tgt) in
                zip(unary_costs_list, gt_perm_mats, lexico_iter(n_points))
            ]

        quadratic_costs_list = [
            # self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]

        if cfg.BBGM.SOLVER_NAME == "LPMP":
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solvers = [
                GraphMatchingModule(
                    all_left_edges,
                    all_right_edges,
                    ns_src,
                    ns_tgt,
                    cfg.BBGM.LAMBDA_VAL,
                    cfg.BBGM.SOLVER_PARAMS,
                )
                for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
                    lexico_iter(all_edges), lexico_iter(n_points)
                )
            ]
            matchings = [
                gm_solver(unary_costs, quadratic_costs)
                for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
            ]
        elif cfg.BBGM.SOLVER_NAME == "LPMP_MGM":
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solver = MultiGraphMatchingModule(
                all_edges, n_points, cfg.BBGM.LAMBDA_VAL, cfg.BBGM.SOLVER_PARAMS)
            matchings = gm_solver(unary_costs_list, quadratic_costs_list)
        else:
            raise ValueError("Unknown solver {}".format(cfg.BBGM.SOLVER_NAME))

        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': None,
                'perm_mat': matchings[0]
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            indices = list(lexico_iter(range(num_graphs)))
            data_dict.update({
                'perm_mat_list': matchings,
                'graph_indices': indices,
                'gt_perm_mat_list': gt_perm_mats
            })

        return data_dict
