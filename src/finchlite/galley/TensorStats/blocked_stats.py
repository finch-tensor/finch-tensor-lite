from collections import OrderedDict
from .tensor_def import TensorDef
from .tensor_stats import TensorStats
from collections.abc import Callable
from typing import Any, Mapping, Self
import numpy as np
from finchlite.finch_logic import Field
import math
import finchlite as fl
class BlockedStats(TensorStats):
    def __init__(self,blocks:np.ndarray,blocks_per_dim:Mapping[Field,int],tensordef: TensorDef, StatsImpl : type[TensorStats]):
        self.blocks = blocks 
        self.blocks_per_dim = blocks_per_dim 
        self.tensordef = tensordef
        self.StatsImpl = StatsImpl

    @classmethod
    def build_grid(cls, d: TensorDef, blocks_per_dim: Mapping[Field, int], StatsImpl: type[TensorStats], data: Any | None = None) -> np.ndarray:
        grid_dim = [blocks_per_dim[idx] for idx in d.index_order]
        blocks_grid = np.empty(grid_dim, dtype=object)

        base_block_sizes = {idx: d.dim_sizes[idx] / blocks_per_dim[idx] for idx in d.index_order}

        for coord in np.ndindex(*grid_dim):
            block_dim_sizes = {}
            slices = []
            for i, idx in enumerate(d.index_order):
                start = math.floor(coord[i] * base_block_sizes[idx])
                end = int(d.dim_sizes[idx] if coord[i] == (blocks_per_dim[idx] - 1) else math.floor((coord[i] + 1) * base_block_sizes[idx]))
                block_dim_sizes[idx] = float(end - start)
                slices.append(slice(start, end))

            if data is not None:
                block_data = data[tuple(slices)].copy()
                wrapped_arr = fl.asarray(block_data)
                local_stats = StatsImpl(wrapped_arr, d.index_order)
                blocks_grid[coord] = local_stats
            else:
                local_def = TensorDef(d.index_order, block_dim_sizes, d.fill_value)
                blocks_grid[coord] = StatsImpl.from_def(local_def)

        return blocks_grid   
    
    @classmethod
    def create_blocked_stats(cls, d : TensorDef, indices : tuple[Field,...], blocks_per_dim : Mapping[Field,int], StatsImpl : type[TensorStats])-> "BlockedStats":
        grid = cls.build_grid(d,blocks_per_dim,StatsImpl,data=None)
        return cls(grid, blocks_per_dim, d.copy(), StatsImpl)
    
    @classmethod
    def from_tensor(cls, tensor : Any, fields : tuple[Field,...], blocks_per_dim = Mapping[Field,int], StatsImpl = type[TensorStats]) -> "BlockedStats" :
        d = TensorDef.from_tensor(tensor,fields)
        data = tensor.to_numpy() if hasattr (tensor,"to_numpy") else tensor
        grid = cls.build_grid(d, blocks_per_dim, StatsImpl, data=data)
        return cls(grid, blocks_per_dim, d, StatsImpl)

    def estimate_non_fill_values(self):
        return float(sum(b.estimate_non_fill_values() for b in self.blocks.flat))


    @staticmethod
    def mapjoin(op : Callable,*args:"BlockedStats") -> "BlockedStats" :
        "We assume that all the args have same sized blocks here"
        def_args = [stat.tensordef for stat in args]
        #For obtaining the tensordef stats for the main block
        new_def = TensorDef.mapjoin(op,*def_args)
        new_blocks = np.empty_like(args[0].blocks)
        InnerStats = args[0].StatsImpl

        for coord in np.ndindex(new_blocks.shape):
            #Obtaining the blocks at the same position in the args
            local_blocks = [arg.blocks[coord] for arg in args]
            new_blocks[coord] = InnerStats.mapjoin(op,*local_blocks)

        return BlockedStats(new_blocks,args[0].blocks_per_dim,new_def,InnerStats)
    
    @staticmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: "TensorStats",
    ) -> "BlockedStats":
        
        new_def = TensorDef.aggregate(op,init,reduce_indices,stats.tensordef)
        #axes we plan to reduce
        grid_reduce_axes = []
        for i,idx in enumerate(stats.index_order):
            if idx in reduce_indices:
                grid_reduce_axes.append(i)
        
        #Defining our new tensor size with squashed dimension size set to 1
        new_grid_shape = list(stats.blocks.shape)
        for axis in grid_reduce_axes:
            new_grid_shape[axis] = 1

        new_blocks = np.empty(new_grid_shape,dtype=object)

        for out_coord in np.ndindex(*new_grid_shape):
            lane_slices = [] 
            for i, val in enumerate(out_coord):
                if i in grid_reduce_axes:
                    lane_slices.append(slice(None))
                else :
                    lane_slices.append(val)

            blocks_in_lane = stats.blocks[tuple(lane_slices)].flat

            #For performing compute over aggregated blocks over the squashed dimension
            lane_accumulator = None
            for b in blocks_in_lane:
                #Perform local aggregate for that block
                local_reduced = stats.StatsImpl.aggregate(op,init,reduce_indices,b)

                if lane_accumulator is None:
                    lane_accumulator = local_reduced
                else :
                    #For blocks that are already aggregated 
                    lane_accumulator = stats.StatsImpl.mapjoin(op,lane_accumulator,local_reduced)

            new_blocks[out_coord] = lane_accumulator

        #Removing the empty dimension
        final_grid = np.squeeze(new_blocks,axis=tuple(grid_reduce_axes))
        #Getting rid of the data associated with squashed diemsnion
        new_blocks_per_dim = {k:v for k,v in stats.blocks_per_dim.items() if k not in reduce_indices}

        return BlockedStats(final_grid,new_blocks_per_dim,new_def, stats.StatsImpl)

    @staticmethod
    def relabel(stats: "BlockedStats", relabel_indices: tuple[Field, ...]) -> "BlockedStats":
        new_def = TensorDef.relabel(stats.tensordef, relabel_indices)
        
        #One to one map of current to changed indices to change the blocks_per_dim data
        name_map = dict(zip(stats.index_order, relabel_indices))
        new_blocks_per_dim = {name_map[k]: v for k, v in stats.blocks_per_dim.items()}
        
        new_blocks = np.empty_like(stats.blocks)
        for coord in np.ndindex(stats.blocks.shape):
            #Relabling every block in the grid
            new_blocks[coord] = stats.StatsImpl.relabel(stats.blocks[coord], relabel_indices)
            
        return BlockedStats(new_blocks, new_blocks_per_dim, new_def, stats.StatsImpl)

    @staticmethod
    def reorder(stats: "BlockedStats", reorder_indices: tuple[Field, ...]) -> "BlockedStats":
        new_def = TensorDef.reorder(stats.tensordef, reorder_indices)
        
        #Mapping existing axes to their new positions in the grid so we can use this to transpose
        old_order = stats.index_order
        axes_mapping = [old_order.index(idx) for idx in reorder_indices if idx in old_order]
        
        #Transposing the data
        new_blocks = np.transpose(stats.blocks, axes=axes_mapping)
        
        #Accounting for any new dummy dimensions added by reorder with size 1
        if len(reorder_indices) > len(old_order):
            expanded_shape = [stats.blocks_per_dim.get(idx, 1) for idx in reorder_indices]
            new_blocks = new_blocks.reshape(expanded_shape)

        #Reordering the blocks inside
        final_blocks = np.empty_like(new_blocks)
        for coord in np.ndindex(new_blocks.shape):
            final_blocks[coord] = stats.StatsImpl.reorder(new_blocks[coord], reorder_indices)
        

        new_blocks_per_dim = {idx: stats.blocks_per_dim.get(idx, 1) for idx in reorder_indices}
        
        return BlockedStats(final_blocks, new_blocks_per_dim, new_def, stats.StatsImpl)
    
    @staticmethod
    def issimilar(a: "BlockedStats", b: "BlockedStats") -> bool:
        if not (isinstance(a, BlockedStats) and isinstance(b, BlockedStats)):
            return False
        
        if a.blocks_per_dim != b.blocks_per_dim or a.StatsImpl != b.StatsImpl:
            return False
        
        # Checking every block
        for block_a, block_b in zip(a.blocks.flat, b.blocks.flat):
            if not a.StatsImpl.issimilar(block_a, block_b):
                return False
        return True
            
    @staticmethod
    def copy_stats(stat: "BlockedStats") -> "BlockedStats":
        if not isinstance(stat, BlockedStats):
            raise TypeError("copy_stats expected a BlockedStats instance")
        
        new_blocks = np.empty_like(stat.blocks)
        
        #Copying every block with it's stats based on the StatsImpl
        for i in range(stat.blocks.size):
            new_blocks.flat[i] = stat.StatsImpl.copy_stats(stat.blocks.flat[i])

        return BlockedStats(
            new_blocks, 
            stat.blocks_per_dim.copy(), 
            stat.tensordef.copy(), 
            stat.StatsImpl
        )
    


        

        




    


        
       